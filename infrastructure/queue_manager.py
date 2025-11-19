"""
Queue Manager - Redis Queue (RQ) Setup for Task Management
Manages multiple priority queues for task distribution
"""

from enum import Enum
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging

try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Provide stub for type hints
    Redis = Any
    Queue = Any
    Job = Any

from core.tdf_schema import TaskDefinition


logger = logging.getLogger(__name__)


class QueuePriority(str, Enum):
    """Queue priority levels"""
    HIGH = "high_priority"
    DEFAULT = "default"
    LOW = "low_priority"


class JobStatus(str, Enum):
    """Job status values from RQ"""
    QUEUED = "queued"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"
    DEFERRED = "deferred"
    SCHEDULED = "scheduled"
    STOPPED = "stopped"
    CANCELED = "canceled"


class QueueManager:
    """
    Manages Redis Queue (RQ) operations for task distribution.

    Implements three priority queues:
    - high_priority: For priority 1-3 tasks
    - default: For priority 4-7 tasks
    - low_priority: For priority 8-10 tasks
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None
    ):
        """
        Initialize the Queue Manager.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Optional Redis password

        Raises:
            ImportError: If redis or rq packages are not installed
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis and RQ are required for QueueManager. "
                "Install them with: pip install redis rq"
            )

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        # Initialize Redis connection
        self.redis_connection = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )

        # Initialize queues
        self.queues: Dict[QueuePriority, Queue] = {
            QueuePriority.HIGH: Queue(
                QueuePriority.HIGH.value,
                connection=self.redis_connection
            ),
            QueuePriority.DEFAULT: Queue(
                QueuePriority.DEFAULT.value,
                connection=self.redis_connection
            ),
            QueuePriority.LOW: Queue(
                QueuePriority.LOW.value,
                connection=self.redis_connection
            )
        }

        # Job tracking
        self.job_registry: Dict[str, str] = {}  # task_id -> job_id mapping

        logger.info(
            f"QueueManager initialized with Redis at {redis_host}:{redis_port}"
        )

    def _get_queue_for_priority(self, priority: int) -> Queue:
        """
        Determine which queue to use based on task priority.

        Args:
            priority: Task priority (1-10)

        Returns:
            Queue object for the appropriate priority level
        """
        if priority <= 3:
            return self.queues[QueuePriority.HIGH]
        elif priority <= 7:
            return self.queues[QueuePriority.DEFAULT]
        else:
            return self.queues[QueuePriority.LOW]

    def enqueue_task(
        self,
        task: TaskDefinition,
        job_func: Optional[Any] = None,
        timeout: Optional[int] = None,
        result_ttl: int = 3600,
        failure_ttl: int = 86400,
        **kwargs
    ) -> str:
        """
        Enqueue a task to the appropriate priority queue.

        Args:
            task: TaskDefinition to enqueue
            job_func: Function to execute (if None, task data is stored for worker pickup)
            timeout: Job timeout in seconds
            result_ttl: How long to keep job results (seconds)
            failure_ttl: How long to keep failed job info (seconds)
            **kwargs: Additional arguments to pass to the job function

        Returns:
            Job ID string

        Raises:
            ValueError: If task priority is invalid
        """
        if task.priority < 1 or task.priority > 10:
            raise ValueError(f"Invalid priority: {task.priority}. Must be 1-10.")

        # Get appropriate queue
        queue = self._get_queue_for_priority(task.priority)

        # Prepare job data
        job_data = task.to_strict_json()

        # If no job_func provided, use a default task processor function
        if job_func is None:
            job_func = self._default_task_processor

        # Enqueue the job
        job = queue.enqueue(
            job_func,
            task_data=job_data,
            timeout=timeout or 600,  # Default 10 minutes
            result_ttl=result_ttl,
            failure_ttl=failure_ttl,
            **kwargs
        )

        # Track the job
        self.job_registry[task.task_id] = job.id

        logger.info(
            f"Enqueued task {task.task_id} to queue {queue.name} "
            f"with job ID {job.id}"
        )

        return job.id

    @staticmethod
    def _default_task_processor(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default task processor function.
        This is a placeholder that workers can override.

        Args:
            task_data: Task data dictionary

        Returns:
            Processing result dictionary
        """
        return {
            "status": "processed",
            "task_id": task_data.get("task_id"),
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get the status of a job.

        Args:
            job_id: Job ID to check

        Returns:
            JobStatus enum value or None if job not found
        """
        try:
            job = Job.fetch(job_id, connection=self.redis_connection)
            return JobStatus(job.get_status())
        except Exception as e:
            logger.warning(f"Could not fetch job {job_id}: {e}")
            return None

    def get_job_result(self, job_id: str) -> Optional[Any]:
        """
        Get the result of a completed job.

        Args:
            job_id: Job ID to get result for

        Returns:
            Job result or None if not available
        """
        try:
            job = Job.fetch(job_id, connection=self.redis_connection)
            return job.result
        except Exception as e:
            logger.warning(f"Could not fetch result for job {job_id}: {e}")
            return None

    def get_job_by_task_id(self, task_id: str) -> Optional[str]:
        """
        Get job ID for a given task ID.

        Args:
            task_id: Task ID to look up

        Returns:
            Job ID or None if not found
        """
        return self.job_registry.get(task_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued or running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if canceled, False otherwise
        """
        try:
            job = Job.fetch(job_id, connection=self.redis_connection)
            job.cancel()
            logger.info(f"Canceled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Could not cancel job {job_id}: {e}")
            return False

    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all queues.

        Returns:
            Dictionary with queue names and their stats
        """
        stats = {}

        for priority, queue in self.queues.items():
            stats[priority.value] = {
                "count": len(queue),
                "started_jobs": queue.started_job_registry.count,
                "finished_jobs": queue.finished_job_registry.count,
                "failed_jobs": queue.failed_job_registry.count,
                "deferred_jobs": queue.deferred_job_registry.count,
                "scheduled_jobs": queue.scheduled_job_registry.count
            }

        return stats

    def clear_queue(self, priority: QueuePriority) -> int:
        """
        Clear all jobs from a specific queue.

        Args:
            priority: Queue priority to clear

        Returns:
            Number of jobs removed
        """
        queue = self.queues[priority]
        count = len(queue)
        queue.empty()
        logger.warning(f"Cleared {count} jobs from {priority.value} queue")
        return count

    def clear_all_queues(self) -> Dict[str, int]:
        """
        Clear all jobs from all queues.

        Returns:
            Dictionary with queue names and number of jobs removed
        """
        result = {}
        for priority in self.queues.keys():
            result[priority.value] = self.clear_queue(priority)
        return result

    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a job.

        Args:
            job_id: Job ID to get info for

        Returns:
            Dictionary with job information or None
        """
        try:
            job = Job.fetch(job_id, connection=self.redis_connection)

            return {
                "job_id": job.id,
                "status": job.get_status(),
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "result": job.result,
                "exc_info": job.exc_info,
                "timeout": job.timeout,
                "origin": job.origin
            }
        except Exception as e:
            logger.warning(f"Could not fetch info for job {job_id}: {e}")
            return None

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.redis_connection.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"QueueManager(host='{self.redis_host}', "
            f"port={self.redis_port}, "
            f"queues={list(self.queues.keys())})"
        )
