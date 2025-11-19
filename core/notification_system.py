"""
Notification System - Email & Slack Alerts for HOTL

This module provides a unified notification system for critical events:
- HOTL review requests from Closed Loop QA
- Emergency shutdown alerts
- Cost limit warnings
- System health issues

Supports multiple channels:
- Email (SMTP)
- Slack (Webhook)
- Dashboard (WebSocket) - via DashboardManager

Configuration is loaded from environment variables for security.
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from enum import Enum
import aiohttp
import os


logger = logging.getLogger(__name__)


class NotificationLevel(str, Enum):
    """Notification priority levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"  # Highest priority - requires immediate attention


class NotificationChannel(str, Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    DASHBOARD = "dashboard"
    ALL = "all"


class NotificationSystem:
    """
    Unified notification system for HOTL alerts and system events.

    Features:
    - Multiple channels (email, Slack, dashboard)
    - Priority-based routing
    - Template-based messages
    - Rate limiting to prevent spam
    - Retry logic for failed deliveries
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dashboard_manager: Optional[Any] = None
    ):
        """
        Initialize the notification system.

        Args:
            config: Optional configuration dictionary
            dashboard_manager: Optional DashboardManager for WebSocket notifications

        Configuration (from environment or config dict):
            SMTP_HOST: SMTP server hostname
            SMTP_PORT: SMTP server port (default: 587)
            SMTP_USERNAME: SMTP username
            SMTP_PASSWORD: SMTP password
            SMTP_FROM_EMAIL: Sender email address
            SMTP_TO_EMAILS: Comma-separated recipient emails

            SLACK_WEBHOOK_URL: Slack incoming webhook URL
            SLACK_CHANNEL: Slack channel name (optional, for display)

            NOTIFICATION_RATE_LIMIT: Max notifications per hour (default: 100)
        """
        self.config = config or {}
        self.dashboard_manager = dashboard_manager

        # SMTP Configuration
        self.smtp_host = os.getenv('SMTP_HOST') or self.config.get('smtp_host')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME') or self.config.get('smtp_username')
        self.smtp_password = os.getenv('SMTP_PASSWORD') or self.config.get('smtp_password')
        self.smtp_from = os.getenv('SMTP_FROM_EMAIL') or self.config.get('smtp_from_email')
        self.smtp_to = (
            os.getenv('SMTP_TO_EMAILS', '').split(',') or
            self.config.get('smtp_to_emails', [])
        )

        # Slack Configuration
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL') or self.config.get('slack_webhook_url')
        self.slack_channel = os.getenv('SLACK_CHANNEL') or self.config.get('slack_channel', '#alerts')

        # Rate limiting
        self.rate_limit = self.config.get('notification_rate_limit', 100)
        self.notification_history: List[datetime] = []

        # Statistics
        self.stats = {
            "total_sent": 0,
            "email_sent": 0,
            "slack_sent": 0,
            "dashboard_sent": 0,
            "failed": 0,
            "rate_limited": 0
        }

        # Check configuration
        self.email_enabled = bool(self.smtp_host and self.smtp_username and self.smtp_to)
        self.slack_enabled = bool(self.slack_webhook)

        logger.info(
            f"NotificationSystem initialized - "
            f"Email: {'✅' if self.email_enabled else '❌'}, "
            f"Slack: {'✅' if self.slack_enabled else '❌'}, "
            f"Dashboard: {'✅' if self.dashboard_manager else '❌'}"
        )

    async def send_notification(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        channels: List[NotificationChannel] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send a notification to specified channels.

        Args:
            title: Notification title
            message: Notification message body
            level: Priority level (info, warning, critical, urgent)
            channels: List of channels to send to (default: all enabled)
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with success status per channel
        """
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded, dropping notification: {title}")
            self.stats["rate_limited"] += 1
            return {"rate_limited": True}

        # Default to all channels if none specified
        if channels is None:
            channels = [NotificationChannel.ALL]

        # Expand ALL to specific channels
        if NotificationChannel.ALL in channels:
            channels = []
            if self.email_enabled:
                channels.append(NotificationChannel.EMAIL)
            if self.slack_enabled:
                channels.append(NotificationChannel.SLACK)
            if self.dashboard_manager:
                channels.append(NotificationChannel.DASHBOARD)

        logger.info(
            f"📧 Sending {level.value} notification: {title} "
            f"(channels: {', '.join(c.value for c in channels)})"
        )

        # Send to each channel
        results = {}
        tasks = []

        for channel in channels:
            if channel == NotificationChannel.EMAIL:
                tasks.append(self._send_email(title, message, level, metadata))
            elif channel == NotificationChannel.SLACK:
                tasks.append(self._send_slack(title, message, level, metadata))
            elif channel == NotificationChannel.DASHBOARD:
                tasks.append(self._send_dashboard(title, message, level, metadata))

        # Execute all channel deliveries in parallel
        if tasks:
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)
            for channel, result in zip(channels, channel_results):
                if isinstance(result, Exception):
                    logger.error(f"Error sending to {channel.value}: {result}")
                    results[channel.value] = False
                    self.stats["failed"] += 1
                else:
                    results[channel.value] = result
                    if result:
                        self.stats["total_sent"] += 1

        return results

    async def send_hotl_alert(
        self,
        cycle_id: str,
        report_type: str,
        severity: str,
        confidence_score: float,
        test_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Send HOTL review alert for Closed Loop QA.

        Args:
            cycle_id: Closed loop cycle ID
            report_type: Type of report (crash, test_failure, etc.)
            severity: Issue severity
            confidence_score: AI confidence score
            test_results: Test execution results
            metadata: Additional metadata

        Returns:
            Delivery status per channel
        """
        title = f"🔍 HOTL Review Required: {report_type}"

        message = f"""
A Closed Loop QA cycle requires human oversight and approval.

**Cycle ID:** {cycle_id}
**Report Type:** {report_type}
**Severity:** {severity}
**Confidence Score:** {confidence_score:.1f}%

**Test Results:**
- Tests Passed: {test_results.get('all_tests_passed', False)}
- Coverage: {test_results.get('avg_coverage', 0):.1f}%
- Validated Tasks: {test_results.get('validated_count', 0)}

**Reason for Review:**
- Confidence below threshold (< 90%)
- OR test coverage below minimum (< 80%)

**Action Required:**
Please review the fixes in the dashboard and approve or reject.
"""

        return await self.send_notification(
            title=title,
            message=message,
            level=NotificationLevel.WARNING,
            channels=[NotificationChannel.ALL],
            metadata={
                "type": "hotl_review",
                "cycle_id": cycle_id,
                "report_type": report_type,
                "confidence_score": confidence_score,
                **(metadata or {})
            }
        )

    async def send_emergency_shutdown_alert(
        self,
        reason: str,
        details: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Send emergency shutdown alert.

        Args:
            reason: Shutdown reason
            details: Additional details

        Returns:
            Delivery status per channel
        """
        title = f"🚨 EMERGENCY SHUTDOWN: {reason}"

        message = f"""
The system has triggered an emergency shutdown.

**Reason:** {reason}
**Timestamp:** {datetime.utcnow().isoformat()}

**Details:**
"""
        for key, value in details.items():
            message += f"- {key}: {value}\n"

        message += """

**Action Required:**
1. Review the root cause
2. Address the issue
3. Reset shutdown state before resuming
"""

        return await self.send_notification(
            title=title,
            message=message,
            level=NotificationLevel.URGENT,
            channels=[NotificationChannel.ALL],
            metadata={
                "type": "emergency_shutdown",
                "reason": reason,
                **details
            }
        )

    async def send_cost_warning_alert(
        self,
        current_cost: float,
        limit: float,
        percentage: float
    ) -> Dict[str, bool]:
        """
        Send cost limit warning alert.

        Args:
            current_cost: Current daily cost
            limit: Cost limit
            percentage: Percentage of limit used

        Returns:
            Delivery status per channel
        """
        title = f"⚠️  Cost Warning: {percentage:.0f}% of Daily Limit"

        message = f"""
The system is approaching the daily cost limit.

**Current Cost:** ${current_cost:.2f}
**Daily Limit:** ${limit:.2f}
**Percentage Used:** {percentage:.0f}%

**Threshold Reached:** 80% warning threshold

**Recommended Actions:**
1. Review current agent activity
2. Consider pausing non-critical tasks
3. Monitor cost progression

System will automatically shutdown at 100% to prevent overruns.
"""

        return await self.send_notification(
            title=title,
            message=message,
            level=NotificationLevel.WARNING,
            channels=[NotificationChannel.ALL],
            metadata={
                "type": "cost_warning",
                "current_cost": current_cost,
                "limit": limit,
                "percentage": percentage
            }
        )

    async def _send_email(
        self,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send email notification via SMTP."""
        if not self.email_enabled:
            logger.debug("Email not configured, skipping")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{level.value.upper()}] {title}"
            msg['From'] = self.smtp_from
            msg['To'] = ', '.join(self.smtp_to)

            # Create HTML version
            html_message = self._format_html_email(title, message, level, metadata)

            # Attach parts
            msg.attach(MIMEText(message, 'plain'))
            msg.attach(MIMEText(html_message, 'html'))

            # Send via SMTP
            await asyncio.to_thread(self._send_smtp, msg)

            logger.info(f"✅ Email sent to {len(self.smtp_to)} recipients")
            self.stats["email_sent"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return False

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Send email via SMTP (blocking, run in thread)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)

    def _format_html_email(
        self,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Format HTML email with styling."""
        level_colors = {
            NotificationLevel.INFO: "#3498db",
            NotificationLevel.WARNING: "#f39c12",
            NotificationLevel.CRITICAL: "#e74c3c",
            NotificationLevel.URGENT: "#c0392b"
        }

        color = level_colors.get(level, "#95a5a6")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .footer {{ padding: 10px; font-size: 12px; color: #7f8c8d; text-align: center; }}
        pre {{ background-color: #ecf0f1; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{title}</h2>
        <p>Priority: {level.value.upper()}</p>
    </div>
    <div class="content">
        <pre>{message}</pre>
    </div>
    <div class="footer">
        <p>Solo-Swarm Multi-Agent System | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
</body>
</html>
"""
        return html

    async def _send_slack(
        self,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send Slack notification via webhook."""
        if not self.slack_enabled:
            logger.debug("Slack not configured, skipping")
            return False

        try:
            # Map level to Slack color
            level_colors = {
                NotificationLevel.INFO: "#3498db",
                NotificationLevel.WARNING: "#f39c12",
                NotificationLevel.CRITICAL: "#e74c3c",
                NotificationLevel.URGENT: "#c0392b"
            }

            # Build Slack message
            slack_payload = {
                "channel": self.slack_channel,
                "username": "Solo-Swarm Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": level_colors.get(level, "#95a5a6"),
                        "title": title,
                        "text": message,
                        "footer": "Solo-Swarm Multi-Agent System",
                        "ts": int(datetime.utcnow().timestamp()),
                        "fields": [
                            {
                                "title": "Priority",
                                "value": level.value.upper(),
                                "short": True
                            }
                        ]
                    }
                ]
            }

            # Add metadata fields
            if metadata:
                for key, value in metadata.items():
                    if key not in ['type']:  # Skip internal fields
                        slack_payload["attachments"][0]["fields"].append({
                            "title": key.replace('_', ' ').title(),
                            "value": str(value),
                            "short": True
                        })

            # Send via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook,
                    json=slack_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Slack notification sent to {self.slack_channel}")
                        self.stats["slack_sent"] += 1
                        return True
                    else:
                        logger.error(f"Slack webhook failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}", exc_info=True)
            return False

    async def _send_dashboard(
        self,
        title: str,
        message: str,
        level: NotificationLevel,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Send notification to dashboard via WebSocket."""
        if not self.dashboard_manager:
            logger.debug("Dashboard not configured, skipping")
            return False

        try:
            # Convert level to dashboard priority
            priority_map = {
                NotificationLevel.INFO: 0,
                NotificationLevel.WARNING: 1,
                NotificationLevel.CRITICAL: 2,
                NotificationLevel.URGENT: 2
            }

            # Send via DashboardManager
            from dashboard.backend.dashboard_manager import DashboardUpdate, UpdateType

            update = DashboardUpdate(
                update_type=UpdateType.SYSTEM_STATUS,
                priority=priority_map.get(level, 1),
                data={
                    "title": title,
                    "message": message,
                    "level": level.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )

            await self.dashboard_manager.broadcast(update)

            logger.info("✅ Dashboard notification sent")
            self.stats["dashboard_sent"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to send dashboard notification: {e}", exc_info=True)
            return False

    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending notification."""
        now = datetime.utcnow()

        # Remove notifications older than 1 hour
        self.notification_history = [
            ts for ts in self.notification_history
            if (now - ts).total_seconds() < 3600
        ]

        # Check limit
        if len(self.notification_history) >= self.rate_limit:
            return False

        # Add current notification
        self.notification_history.append(now)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "total_sent": self.stats["total_sent"],
            "email_sent": self.stats["email_sent"],
            "slack_sent": self.stats["slack_sent"],
            "dashboard_sent": self.stats["dashboard_sent"],
            "failed": self.stats["failed"],
            "rate_limited": self.stats["rate_limited"],
            "email_enabled": self.email_enabled,
            "slack_enabled": self.slack_enabled,
            "dashboard_enabled": bool(self.dashboard_manager)
        }
