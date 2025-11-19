/**
 * Solo-Swarm Dashboard - Main Application
 *
 * Features:
 * - WebSocket connection with auto-reconnect
 * - Real-time updates for agents, costs, tasks
 * - Interactive approval queue
 * - Live activity log
 * - Agent grid with tooltips
 */

// Configuration
const CONFIG = {
    // WebSocket URL - adjust based on your backend
    WS_URL: 'ws://localhost:8000/ws/dashboard',

    // API Base URL
    API_URL: 'http://localhost:8000/api',

    // Reconnect settings
    RECONNECT_DELAY: 3000,
    MAX_RECONNECT_ATTEMPTS: 10,

    // Cost budget (in USD)
    DAILY_BUDGET: 100.00,
    WARNING_THRESHOLD: 0.8  // 80%
};

// Application State
const state = {
    ws: null,
    reconnectAttempts: 0,
    connected: false,
    agents: Array(100).fill(null).map((_, i) => ({
        slot_id: i,
        status: 'idle',
        current_task: null,
        agent_type: null
    })),
    costs: {
        total_cost_usd: 0,
        total_tokens: 0
    },
    stats: {
        tasks: 0,
        completed: 0,
        in_progress: 0,
        failed: 0,
        validations: 0,
        pass_rate: 0
    },
    approvals: []
};

// ===== INITIALIZATION =====

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Solo-Swarm Dashboard initialized');

    // Initialize components
    initializeAgentGrid();
    connectWebSocket();
    loadInitialData();
    setupEventListeners();

    // Add initial log
    addLog('system', 'Dashboard loaded and ready');
});

// ===== WEBSOCKET CONNECTION =====

function connectWebSocket() {
    addLog('system', `Connecting to ${CONFIG.WS_URL}...`);

    try {
        state.ws = new WebSocket(CONFIG.WS_URL);

        state.ws.onopen = handleWebSocketOpen;
        state.ws.onmessage = handleWebSocketMessage;
        state.ws.onclose = handleWebSocketClose;
        state.ws.onerror = handleWebSocketError;

    } catch (error) {
        console.error('WebSocket connection error:', error);
        addLog('error', `Connection failed: ${error.message}`);
        scheduleReconnect();
    }
}

function handleWebSocketOpen() {
    console.log('✅ WebSocket connected');
    state.connected = true;
    state.reconnectAttempts = 0;

    updateConnectionStatus(true);
    addLog('success', 'Connected to backend');
}

function handleWebSocketMessage(event) {
    try {
        const update = JSON.parse(event.data);
        console.log('📨 Received update:', update);

        // Route update to appropriate handler
        switch (update.update_type) {
            case 'agent_status':
                handleAgentStatusUpdate(update.data);
                break;

            case 'task_update':
                handleTaskUpdate(update.data);
                break;

            case 'cost_update':
                handleCostUpdate(update.data);
                break;

            case 'validation_result':
                handleValidationResult(update.data);
                break;

            case 'approval_request':
                handleApprovalRequest(update.data);
                break;

            case 'system_status':
                handleSystemStatus(update.data);
                break;

            default:
                console.warn('Unknown update type:', update.update_type);
        }

    } catch (error) {
        console.error('Error processing WebSocket message:', error);
        addLog('error', `Message processing error: ${error.message}`);
    }
}

function handleWebSocketClose(event) {
    console.log('❌ WebSocket closed:', event);
    state.connected = false;

    updateConnectionStatus(false);
    addLog('warning', 'Connection lost');

    scheduleReconnect();
}

function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    addLog('error', 'Connection error occurred');
}

function scheduleReconnect() {
    if (state.reconnectAttempts >= CONFIG.MAX_RECONNECT_ATTEMPTS) {
        addLog('error', 'Max reconnection attempts reached. Please refresh.');
        return;
    }

    state.reconnectAttempts++;

    addLog('system', `Reconnecting in ${CONFIG.RECONNECT_DELAY / 1000}s (attempt ${state.reconnectAttempts})...`);

    setTimeout(() => {
        connectWebSocket();
    }, CONFIG.RECONNECT_DELAY);
}

function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');

    if (connected) {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
        statusText.style.color = 'var(--status-idle)';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
        statusText.style.color = 'var(--status-error)';
    }
}

// ===== UPDATE HANDLERS =====

function handleAgentStatusUpdate(data) {
    const { agent_id, status, details } = data;

    // Parse agent slot from agent_id (e.g., "agent_001" -> 1)
    const slotMatch = agent_id.match(/\d+/);
    if (!slotMatch) return;

    const slot = parseInt(slotMatch[0]);
    if (slot >= 0 && slot < 100) {
        state.agents[slot] = {
            slot_id: slot,
            status: status,
            current_task: details?.current_task || null,
            agent_type: details?.agent_type || null
        };

        updateAgentSlot(slot);
        updateAgentCount();

        addLog('system', `Agent #${slot}: ${status}`);
    }
}

function handleTaskUpdate(data) {
    const { task_id, status, progress, details } = data;

    addLog('system', `Task ${task_id}: ${status}` + (progress ? ` (${progress}%)` : ''));

    // Update stats (will be refreshed with next stats call)
    refreshStats();
}

function handleCostUpdate(data) {
    const { total_cost_usd, total_tokens, breakdown } = data;

    state.costs = {
        total_cost_usd,
        total_tokens
    };

    updateCostMonitor();
    addLog('system', `Cost update: $${total_cost_usd.toFixed(2)}`);
}

function handleValidationResult(data) {
    const { asset_path, is_valid, issues } = data;

    const status = is_valid ? 'success' : 'error';
    const icon = is_valid ? '✓' : '✗';

    addLog(status, `${icon} Validation: ${asset_path.split('/').pop()}`);

    if (!is_valid && issues.length > 0) {
        addLog('warning', `  Issues: ${issues.join(', ')}`);
    }

    refreshStats();
}

function handleApprovalRequest(data) {
    const { task_id, task_type, description, metadata } = data;

    // Add to approval queue
    state.approvals.push({
        task_id,
        task_type,
        description,
        metadata
    });

    updateApprovalQueue();
    addLog('warning', `⚠ Approval requested: ${task_id}`);
}

function handleSystemStatus(data) {
    const { status, message } = data;
    addLog('system', message);
}

// ===== AGENT GRID =====

function initializeAgentGrid() {
    const grid = document.getElementById('agent-grid');

    for (let i = 0; i < 100; i++) {
        const slot = document.createElement('div');
        slot.className = 'agent-slot idle';
        slot.dataset.slot = i;

        // Tooltip on hover
        slot.addEventListener('mouseenter', (e) => showAgentTooltip(e, i));
        slot.addEventListener('mouseleave', hideAgentTooltip);
        slot.addEventListener('mousemove', moveTooltip);

        grid.appendChild(slot);
    }
}

function updateAgentSlot(slotId) {
    const slot = document.querySelector(`[data-slot="${slotId}"]`);
    if (!slot) return;

    const agent = state.agents[slotId];

    // Remove all status classes
    slot.classList.remove('idle', 'busy', 'thinking', 'error');

    // Add current status class
    slot.classList.add(agent.status);
}

function updateAgentCount() {
    const busyCount = state.agents.filter(a => a.status === 'busy' || a.status === 'thinking').length;
    document.getElementById('agent-count').textContent = `${busyCount}/100`;
}

function showAgentTooltip(event, slotId) {
    const tooltip = document.getElementById('tooltip');
    const agent = state.agents[slotId];

    let content = `<strong>Slot #${slotId}</strong><br>`;
    content += `Status: ${agent.status}<br>`;

    if (agent.current_task) {
        content += `Task: ${agent.current_task}<br>`;
    }

    if (agent.agent_type) {
        content += `Type: ${agent.agent_type}`;
    }

    tooltip.querySelector('.tooltip-content').innerHTML = content;
    tooltip.style.display = 'block';

    moveTooltip(event);
}

function hideAgentTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

function moveTooltip(event) {
    const tooltip = document.getElementById('tooltip');
    tooltip.style.left = (event.pageX + 15) + 'px';
    tooltip.style.top = (event.pageY + 15) + 'px';
}

// ===== COST MONITOR =====

function updateCostMonitor() {
    const { total_cost_usd, total_tokens } = state.costs;

    // Update values
    document.getElementById('cost-value').textContent = `$${total_cost_usd.toFixed(2)}`;
    document.getElementById('cost-tokens').textContent = `${total_tokens.toLocaleString()} tokens`;

    // Update progress bar
    const percentage = (total_cost_usd / CONFIG.DAILY_BUDGET) * 100;
    const fill = document.getElementById('cost-bar-fill');
    fill.style.width = `${Math.min(percentage, 100)}%`;

    // Update warning states
    const monitor = document.getElementById('cost-monitor');
    monitor.classList.remove('warning', 'danger');
    fill.classList.remove('warning', 'danger');

    if (percentage >= 100) {
        monitor.classList.add('danger');
        fill.classList.add('danger');
        addLog('error', '🚨 Daily budget exceeded!');
    } else if (percentage >= CONFIG.WARNING_THRESHOLD * 100) {
        monitor.classList.add('warning');
        fill.classList.add('warning');
    }
}

// ===== APPROVAL QUEUE =====

function updateApprovalQueue() {
    const queue = document.getElementById('approval-queue');
    const count = document.getElementById('approval-count');

    count.textContent = state.approvals.length;

    if (state.approvals.length === 0) {
        queue.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">✓</div>
                <p>No pending approvals</p>
            </div>
        `;
        return;
    }

    queue.innerHTML = '';

    state.approvals.forEach(approval => {
        const card = createApprovalCard(approval);
        queue.appendChild(card);
    });
}

function createApprovalCard(approval) {
    const template = document.getElementById('approval-card-template');
    const card = template.content.cloneNode(true).querySelector('.approval-card');

    card.dataset.taskId = approval.task_id || approval.cycle_id;
    card.querySelector('.approval-type').textContent = approval.task_type || approval.report_type || 'REVIEW';
    card.querySelector('.approval-priority').textContent = approval.severity || 'HIGH';
    card.querySelector('.approval-title').textContent = approval.cycle_id || approval.task_id;
    card.querySelector('.approval-description').textContent =
        approval.description || `Closed Loop cycle requires manual review (confidence: ${approval.confidence_score?.toFixed(1)}%)`;

    // Metadata
    if (approval.metadata) {
        const metadataText = Object.entries(approval.metadata)
            .map(([k, v]) => `${k}: ${v}`)
            .join(' | ');
        card.querySelector('.approval-metadata').textContent = metadataText;
    } else if (approval.test_results) {
        // Show summary for closed loop approvals
        const testResults = approval.test_results;
        card.querySelector('.approval-metadata').textContent =
            `Tests: ${testResults.all_tests_passed ? '✅' : '❌'} | Coverage: ${testResults.avg_coverage?.toFixed(1)}%`;
    }

    // Click card to open modal with full details
    card.style.cursor = 'pointer';
    card.addEventListener('click', (e) => {
        // Don't open modal if clicking buttons
        if (e.target.closest('.btn-approve') || e.target.closest('.btn-reject')) {
            return;
        }
        openApprovalModal(approval);
    });

    // Button handlers - for quick approval without opening modal
    card.querySelector('.btn-approve').addEventListener('click', (e) => {
        e.stopPropagation();
        handleApproval(approval.task_id || approval.cycle_id, true);
    });

    card.querySelector('.btn-reject').addEventListener('click', (e) => {
        e.stopPropagation();
        handleApproval(approval.task_id || approval.cycle_id, false);
    });

    return card;
}

async function handleApproval(taskId, approved) {
    try {
        const response = await fetch(`${CONFIG.API_URL}/approval/${taskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                approved: approved,
                approved_by: 'dashboard_user',
                comment: approved ? 'Approved via dashboard' : 'Rejected via dashboard'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Remove from queue
        state.approvals = state.approvals.filter(a => a.task_id !== taskId);
        updateApprovalQueue();

        const action = approved ? 'approved' : 'rejected';
        addLog(approved ? 'success' : 'warning', `Task ${taskId} ${action}`);

    } catch (error) {
        console.error('Approval error:', error);
        addLog('error', `Failed to ${approved ? 'approve' : 'reject'} task: ${error.message}`);
    }
}

// ===== STATS =====

function updateStats(stats) {
    state.stats = stats;

    document.getElementById('stat-tasks').textContent = stats.tasks || 0;
    document.getElementById('stat-completed').textContent = stats.completed || 0;
    document.getElementById('stat-in-progress').textContent = stats.in_progress || 0;
    document.getElementById('stat-failed').textContent = stats.failed || 0;
    document.getElementById('stat-validations').textContent = stats.validations || 0;
    document.getElementById('stat-pass-rate').textContent = `${stats.pass_rate || 0}%`;
}

async function refreshStats() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/stats`);
        if (!response.ok) return;

        const data = await response.json();

        // Process stats
        const tasks = data.tasks || {};
        const stats = {
            tasks: Object.values(tasks).reduce((a, b) => a + b, 0),
            completed: tasks.completed || 0,
            in_progress: tasks.in_progress || 0,
            failed: tasks.failed || 0,
            validations: (data.validations?.passed || 0) + (data.validations?.failed || 0),
            pass_rate: data.validations?.passed && data.validations?.failed ?
                Math.round((data.validations.passed / (data.validations.passed + data.validations.failed)) * 100) : 0
        };

        updateStats(stats);

    } catch (error) {
        console.error('Failed to refresh stats:', error);
    }
}

// ===== ACTIVITY LOG =====

function addLog(type, message) {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const now = new Date();
    const time = now.toTimeString().split(' ')[0];

    const icons = {
        system: 'ℹ',
        success: '✓',
        warning: '⚠',
        error: '✗'
    };

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-icon">${icons[type] || 'ℹ'}</span>
        <span class="log-message">${message}</span>
    `;

    log.appendChild(entry);

    // Auto-scroll to bottom
    log.scrollTop = log.scrollHeight;

    // Keep only last 100 entries
    while (log.children.length > 100) {
        log.removeChild(log.firstChild);
    }
}

// ===== DATA LOADING =====

async function loadInitialData() {
    try {
        // Load agent status
        const agentsResponse = await fetch(`${CONFIG.API_URL}/status/agents`);
        if (agentsResponse.ok) {
            const agents = await agentsResponse.json();
            agents.forEach(agent => {
                state.agents[agent.slot_id] = agent;
                updateAgentSlot(agent.slot_id);
            });
            updateAgentCount();
            addLog('success', `Loaded ${agents.length} agent slots`);
        }

        // Load costs
        const costsResponse = await fetch(`${CONFIG.API_URL}/costs/today`);
        if (costsResponse.ok) {
            const costs = await costsResponse.json();
            state.costs = {
                total_cost_usd: costs.total_cost_usd,
                total_tokens: costs.total_tokens
            };
            updateCostMonitor();
            addLog('success', `Loaded cost data: $${costs.total_cost_usd.toFixed(2)}`);
        }

        // Load stats
        await refreshStats();

    } catch (error) {
        console.error('Failed to load initial data:', error);
        addLog('warning', 'Failed to load some data, will retry...');
    }
}

// ===== EVENT LISTENERS =====

function setupEventListeners() {
    // Clear log button
    document.getElementById('clear-log').addEventListener('click', () => {
        const log = document.getElementById('activity-log');
        log.innerHTML = '';
        addLog('system', 'Log cleared');
    });

    // Modal close button
    document.getElementById('modal-close').addEventListener('click', closeApprovalModal);

    // Modal overlay click to close
    document.querySelector('.modal-overlay').addEventListener('click', closeApprovalModal);

    // Test output collapsible toggle
    document.getElementById('test-output-toggle').addEventListener('click', function() {
        const output = document.getElementById('modal-test-output');
        const isHidden = output.style.display === 'none';
        output.style.display = isHidden ? 'block' : 'none';
        this.classList.toggle('open', isHidden);
    });

    // Modal approve button
    document.getElementById('modal-approve-btn').addEventListener('click', handleModalApprove);

    // Modal reject button
    document.getElementById('modal-reject-btn').addEventListener('click', handleModalReject);

    // Refresh data periodically
    setInterval(refreshStats, 30000); // Every 30 seconds
}

// ===== APPROVAL MODAL =====

let currentApprovalData = null;

function openApprovalModal(approvalData) {
    currentApprovalData = approvalData;
    const modal = document.getElementById('approval-modal');

    // Populate modal with data
    document.getElementById('modal-cycle-id').textContent = approvalData.cycle_id || '-';
    document.getElementById('modal-report-type').textContent = approvalData.report_type || '-';
    document.getElementById('modal-severity').textContent = approvalData.severity || '-';
    document.getElementById('modal-confidence').textContent =
        approvalData.confidence_score ? `${approvalData.confidence_score.toFixed(1)}%` : '-';

    // Test results
    const testResults = approvalData.test_results || {};
    document.getElementById('modal-tests-passed').textContent =
        testResults.all_tests_passed ? '✅ Yes' : '❌ No';
    document.getElementById('modal-coverage').textContent =
        testResults.avg_coverage ? `${testResults.avg_coverage.toFixed(1)}%` : '-';
    document.getElementById('modal-iterations').textContent =
        testResults.iterations || '1';

    // Changes
    const changesEl = document.getElementById('modal-changes');
    if (approvalData.changes && approvalData.changes.length > 0) {
        const changesList = document.createElement('ul');
        approvalData.changes.forEach(file => {
            const li = document.createElement('li');
            li.textContent = file;
            changesList.appendChild(li);
        });
        changesEl.innerHTML = '';
        changesEl.appendChild(changesList);
    } else {
        changesEl.innerHTML = '<p class="text-muted">No changes detected</p>';
    }

    // Reason for review
    const reasonList = document.getElementById('modal-reason-list');
    reasonList.innerHTML = '';

    const reasons = [];
    if (approvalData.confidence_score < 90) {
        reasons.push(`Confidence score (${approvalData.confidence_score.toFixed(1)}%) is below threshold (90%)`);
    }
    if (testResults.avg_coverage < 80) {
        reasons.push(`Test coverage (${testResults.avg_coverage?.toFixed(1)}%) is below minimum (80%)`);
    }
    if (!testResults.all_tests_passed) {
        reasons.push('Some tests failed during validation');
    }

    reasons.forEach(reason => {
        const li = document.createElement('li');
        li.textContent = reason;
        reasonList.appendChild(li);
    });

    // Test output
    const testOutput = approvalData.test_output || testResults.test_output_snippet || 'No test output available';
    document.getElementById('modal-test-output-code').textContent = testOutput;

    // Show modal
    modal.style.display = 'flex';
    addLog('info', `Opened approval modal for cycle ${approvalData.cycle_id}`);
}

function closeApprovalModal() {
    const modal = document.getElementById('approval-modal');
    modal.style.display = 'none';
    currentApprovalData = null;

    // Reset collapsible
    const output = document.getElementById('modal-test-output');
    const toggle = document.getElementById('test-output-toggle');
    output.style.display = 'none';
    toggle.classList.remove('open');
}

async function handleModalApprove() {
    if (!currentApprovalData) return;

    const cycleId = currentApprovalData.cycle_id;
    const taskId = currentApprovalData.task_id || cycleId;

    try {
        addLog('info', `Approving cycle ${cycleId}...`);

        const response = await fetch(`${CONFIG.API_URL}/approval/${taskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'approve',
                cycle_id: cycleId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        addLog('success', `✅ Cycle ${cycleId} approved and merged`);
        closeApprovalModal();

        // Remove from approval queue
        removeApprovalFromQueue(taskId);

    } catch (error) {
        console.error('Error approving cycle:', error);
        addLog('error', `Failed to approve cycle: ${error.message}`);
        alert(`Failed to approve: ${error.message}`);
    }
}

async function handleModalReject() {
    if (!currentApprovalData) return;

    const cycleId = currentApprovalData.cycle_id;
    const taskId = currentApprovalData.task_id || cycleId;

    try {
        addLog('warning', `Rejecting cycle ${cycleId}...`);

        const response = await fetch(`${CONFIG.API_URL}/approval/${taskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'reject',
                cycle_id: cycleId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        addLog('warning', `❌ Cycle ${cycleId} rejected`);
        closeApprovalModal();

        // Remove from approval queue
        removeApprovalFromQueue(taskId);

    } catch (error) {
        console.error('Error rejecting cycle:', error);
        addLog('error', `Failed to reject cycle: ${error.message}`);
        alert(`Failed to reject: ${error.message}`);
    }
}

function removeApprovalFromQueue(taskId) {
    // Find and remove the approval card
    const card = document.querySelector(`.approval-card[data-task-id="${taskId}"]`);
    if (card) {
        card.remove();
    }

    // Update state
    state.approvals = state.approvals.filter(a => (a.task_id || a.cycle_id) !== taskId);

    // Update count
    updateApprovalCount();

    // Show empty state if no approvals left
    if (state.approvals.length === 0) {
        showEmptyApprovalState();
    }
}

function updateApprovalCount() {
    document.getElementById('approval-count').textContent = state.approvals.length;
}

function showEmptyApprovalState() {
    const queue = document.getElementById('approval-queue');
    queue.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">✓</div>
            <p>No pending approvals</p>
        </div>
    `;
}

// ===== EXPORTS (for testing) =====

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CONFIG,
        state,
        addLog,
        updateCostMonitor,
        handleAgentStatusUpdate
    };
}

// ========================================
// BRAIN UI - GENETIC MEMORY INTERFACE
// ========================================

// Brain UI State
const brainState = {
    currentAgentType: 'architect_agent',
    versionHistory: [],
    performanceData: null,
    selectedVersions: [],
    chartInstance: null
};

// ========================================
// TAB SWITCHING
// ========================================

function initTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const views = document.querySelectorAll('.view-container');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            // Update active states
            tabButtons.forEach(b => b.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));

            btn.classList.add('active');
            const targetView = document.getElementById('view-' + targetTab);
            if (targetView) {
                targetView.classList.add('active');
            }

            // Load brain data if switching to brain tab
            if (targetTab === 'brain') {
                loadBrainData();
            }

            addLog('Switched to ' + targetTab + ' view', 'system');
        });
    });
}

// ========================================
// BRAIN DATA LOADING
// ========================================

async function loadBrainData() {
    const agentType = brainState.currentAgentType;
    addLog('Loading Brain data for ' + agentType + '...', 'system');

    try {
        // Load version history
        await loadVersionHistory(agentType);

        // Load performance comparison
        await loadPerformanceComparison(agentType);

        addLog('Brain data loaded for ' + agentType, 'success');
    } catch (error) {
        console.error('Error loading brain data:', error);
        addLog('Failed to load Brain data: ' + error.message, 'error');
    }
}

async function loadVersionHistory(agentType, limit = 20) {
    try {
        const response = await fetch(
            CONFIG.API_URL + '/prompts/' + agentType + '/history?limit=' + limit
        );

        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }

        const data = await response.json();
        brainState.versionHistory = data.versions || [];

        renderVersionTimeline();
        renderPerformanceChart();
    } catch (error) {
        console.error('Error loading version history:', error);
        brainState.versionHistory = [];
        renderVersionTimeline();
        throw error;
    }
}

async function loadPerformanceComparison(agentType) {
    try {
        const response = await fetch(
            CONFIG.API_URL + '/prompts/' + agentType + '/comparison'
        );

        if (!response.ok) {
            throw new Error('HTTP ' + response.status);
        }

        const data = await response.json();
        brainState.performanceData = data;

        updateOverviewCards(data);
    } catch (error) {
        console.error('Error loading performance comparison:', error);
        brainState.performanceData = null;
        updateOverviewCards(null);
        throw error;
    }
}

// ========================================
// UI RENDERING
// ========================================

function updateOverviewCards(data) {
    if (!data || data.error) {
        document.getElementById('brain-current-version').textContent = '-';
        document.getElementById('brain-current-score').textContent = '-';
        document.getElementById('brain-best-version').textContent = '-';
        document.getElementById('brain-best-score').textContent = '-';
        document.getElementById('brain-total-versions').textContent = '-';
        document.getElementById('brain-trend').textContent = '-';
        document.getElementById('brain-avg-score').textContent = '-';
        document.getElementById('brain-score-range').textContent = '-';
        return;
    }

    // Current Version
    const current = data.current_version || {};
    document.getElementById('brain-current-version').textContent =
        current.version ? 'v' + current.version : '-';
    document.getElementById('brain-current-score').textContent =
        current.score ? 'Score: ' + current.score.toFixed(1) : '-';

    // Best Version
    const best = data.best_version || {};
    document.getElementById('brain-best-version').textContent =
        best.version ? 'v' + best.version : '-';
    document.getElementById('brain-best-score').textContent =
        best.score ? 'Score: ' + best.score.toFixed(1) : '-';

    // Statistics
    const stats = data.statistics || {};
    document.getElementById('brain-total-versions').textContent =
        stats.total_versions || '-';
    document.getElementById('brain-avg-score').textContent =
        stats.avg_score ? stats.avg_score.toFixed(1) : '-';
    document.getElementById('brain-score-range').textContent =
        stats.score_range ? 'Range: ' + stats.score_range.toFixed(1) : '-';

    // Trend
    const trend = data.recent_trend || '-';
    const trendEmoji = {
        'improving': '📈 Improving',
        'degrading': '📉 Degrading',
        'stable': '➡️ Stable',
        'insufficient_data': '❓ Insufficient Data'
    };
    document.getElementById('brain-trend').textContent =
        trendEmoji[trend] || trend;
}

function renderVersionTimeline() {
    const timeline = document.getElementById('version-timeline');
    const template = document.getElementById('version-card-template');

    if (!brainState.versionHistory || brainState.versionHistory.length === 0) {
        timeline.innerHTML = '<div class="empty-state"><div class="empty-icon">🧬</div><p>No version history available</p><small>This agent hasn not been optimized yet</small></div>';
        return;
    }

    timeline.innerHTML = '';

    brainState.versionHistory.forEach(version => {
        const card = template.content.cloneNode(true);
        const cardDiv = card.querySelector('.version-card');

        cardDiv.dataset.version = version.version;
        if (version.is_active) {
            cardDiv.classList.add('active');
        }

        // Version number
        card.querySelector('.version-num').textContent = version.version;

        // Status
        const status = card.querySelector('.version-status');
        status.textContent = version.is_active ? 'Active' : 'Inactive';
        status.classList.add(version.is_active ? 'active' : 'inactive');

        // Metadata
        const date = new Date(version.created_at || version.activated_at);
        card.querySelector('.version-date').textContent =
            date.toLocaleDateString();
        card.querySelector('.version-author').textContent =
            version.changed_by || 'Unknown';

        // Reason
        card.querySelector('.version-reason').textContent =
            version.change_reason || 'No reason provided';

        // Metrics
        card.querySelector('.metric-value.score').textContent =
            version.performance_score ? version.performance_score.toFixed(1) : 'N/A';
        card.querySelector('.metric-value.success').textContent =
            version.success_rate ? version.success_rate.toFixed(1) + '%' : 'N/A';
        card.querySelector('.metric-value.cost').textContent =
            version.avg_cost ? '$' + version.avg_cost.toFixed(3) : 'N/A';

        // Actions
        card.querySelector('.btn-view').addEventListener('click', (e) => {
            e.stopPropagation();
            openBrainModal(version);
        });

        const rollbackBtn = card.querySelector('.btn-rollback');
        if (!version.is_active) {
            rollbackBtn.style.display = 'flex';
            rollbackBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                confirmRollback(version);
            });
        }

        // Card click to view
        cardDiv.addEventListener('click', () => {
            openBrainModal(version);
        });

        timeline.appendChild(card);
    });
}

function renderPerformanceChart() {
    const canvas = document.getElementById('perf-chart-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const versions = brainState.versionHistory;

    if (!versions || versions.length === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6b7280';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No performance data available', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Extract data
    const versionsWithScores = versions
        .filter(v => v.performance_score !== null)
        .reverse(); // Oldest first

    if (versionsWithScores.length === 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6b7280';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No performance scores recorded', canvas.width / 2, canvas.height / 2);
        return;
    }

    const scores = versionsWithScores.map(v => v.performance_score);
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    const scoreRange = maxScore - minScore || 1;

    // Chart dimensions
    const padding = 40;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight * i / 5);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();

        // Y-axis labels
        const scoreValue = maxScore - (scoreRange * i / 5);
        ctx.fillStyle = '#9ca3af';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(scoreValue.toFixed(1), padding - 10, y + 4);
    }

    // Draw line chart
    ctx.strokeStyle = '#00d9ff';
    ctx.lineWidth = 3;
    ctx.beginPath();

    versionsWithScores.forEach((version, index) => {
        const x = padding + (chartWidth * index / (versionsWithScores.length - 1 || 1));
        const normalizedScore = (version.performance_score - minScore) / scoreRange;
        const y = padding + chartHeight * (1 - normalizedScore);

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    // Draw data points
    versionsWithScores.forEach((version, index) => {
        const x = padding + (chartWidth * index / (versionsWithScores.length - 1 || 1));
        const normalizedScore = (version.performance_score - minScore) / scoreRange;
        const y = padding + chartHeight * (1 - normalizedScore);

        // Point circle
        ctx.fillStyle = version.is_active ? '#00ff88' : '#00d9ff';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Version label
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('v' + version.version, x, canvas.height - padding + 15);
    });

    // Chart title
    ctx.fillStyle = '#f3f4f6';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Performance Score Evolution', canvas.width / 2, 20);
}

// ========================================
// BRAIN MODAL
// ========================================

function openBrainModal(version) {
    const modal = document.getElementById('brain-modal');

    // Populate version info
    document.getElementById('brain-version-num').textContent = 'v' + version.version;
    document.getElementById('brain-agent-type').textContent = brainState.currentAgentType;

    const statusBadge = document.getElementById('brain-status');
    statusBadge.textContent = version.is_active ? 'Active' : 'Inactive';
    statusBadge.className = 'info-value badge ' + (version.is_active ? 'success' : 'secondary');

    document.getElementById('brain-changed-by').textContent = version.changed_by || 'Unknown';

    // Performance metrics
    document.getElementById('brain-perf-score').textContent =
        version.performance_score ? version.performance_score.toFixed(2) : 'N/A';
    document.getElementById('brain-success-rate').textContent =
        version.success_rate ? version.success_rate.toFixed(1) + '%' : 'N/A';
    document.getElementById('brain-avg-cost').textContent =
        version.avg_cost ? '$' + version.avg_cost.toFixed(4) : 'N/A';
    document.getElementById('brain-avg-duration').textContent =
        version.avg_duration ? version.avg_duration.toFixed(1) + 's' : 'N/A';

    // Change reason
    document.getElementById('brain-change-reason').innerHTML =
        '<p>' + (version.change_reason || 'No reason provided') + '</p>';

    // Prompt content
    document.getElementById('brain-prompt-code').textContent =
        version.content || 'No content available';

    // Shadow test results
    const shadowSection = document.getElementById('brain-shadow-section');
    if (version.shadow_test_count) {
        shadowSection.style.display = 'block';
        document.getElementById('brain-shadow-count').textContent = version.shadow_test_count;
        document.getElementById('brain-shadow-rate').textContent =
            version.shadow_test_success_rate ? version.shadow_test_success_rate.toFixed(1) + '%' : 'N/A';
    } else {
        shadowSection.style.display = 'none';
    }

    // Rollback info
    const rollbackSection = document.getElementById('brain-rollback-section');
    if (version.metadata && version.metadata.last_rollback) {
        rollbackSection.style.display = 'block';
        const rollback = version.metadata.last_rollback;
        document.getElementById('brain-rollback-text').textContent =
            'Rolled back from v' + rollback.from_version + ': ' + rollback.reason;
    } else {
        rollbackSection.style.display = 'none';
    }

    // Rollback button
    const rollbackBtn = document.getElementById('brain-rollback-btn');
    if (!version.is_active) {
        rollbackBtn.style.display = 'flex';
        rollbackBtn.onclick = () => {
            modal.style.display = 'none';
            confirmRollback(version);
        };
    } else {
        rollbackBtn.style.display = 'none';
    }

    modal.style.display = 'flex';
    addLog('Viewing version ' + version.version + ' details', 'info');
}

function closeBrainModal() {
    document.getElementById('brain-modal').style.display = 'none';
}

// ========================================
// ROLLBACK FUNCTIONALITY
// ========================================

async function confirmRollback(version) {
    const confirmed = confirm(
        'Are you sure you want to rollback to version ' + version.version + '?\n\n' +
        'This will deactivate the current version and reactivate v' + version.version + '.\n' +
        'Reason: ' + version.change_reason
    );

    if (!confirmed) return;

    const reason = prompt('Enter rollback reason:', 'Manual rollback via dashboard');
    if (!reason) return;

    addLog('Rolling back to version ' + version.version + '...', 'system');

    try {
        const response = await fetch(
            CONFIG.API_URL + '/prompts/' + brainState.currentAgentType + '/rollback',
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_version: version.version,
                    reason: reason
                })
            }
        );

        if (!response.ok) {
            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
        }

        const result = await response.json();
        addLog('✅ Successfully rolled back to version ' + version.version, 'success');

        // Reload brain data
        await loadBrainData();
    } catch (error) {
        console.error('Rollback error:', error);
        addLog('❌ Rollback failed: ' + error.message, 'error');
        alert('Rollback failed: ' + error.message);
    }
}

// ========================================
// EVENT HANDLERS
// ========================================

function initBrainEventHandlers() {
    // Agent type selector
    const agentSelect = document.getElementById('agent-type-select');
    if (agentSelect) {
        agentSelect.addEventListener('change', (e) => {
            brainState.currentAgentType = e.target.value;
            loadBrainData();
        });
    }

    // Refresh button
    const refreshBtn = document.getElementById('refresh-brain-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            loadBrainData();
        });
    }

    // Brain modal close button
    const brainModalClose = document.getElementById('brain-modal-close');
    if (brainModalClose) {
        brainModalClose.addEventListener('click', closeBrainModal);
    }

    const brainCloseBtn = document.getElementById('brain-close-btn');
    if (brainCloseBtn) {
        brainCloseBtn.addEventListener('click', closeBrainModal);
    }

    // Prompt content collapsible
    const promptToggle = document.getElementById('brain-prompt-toggle');
    if (promptToggle) {
        promptToggle.addEventListener('click', () => {
            const content = document.getElementById('brain-prompt-content');
            const icon = promptToggle.querySelector('.toggle-icon');

            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.textContent = '▲';
            } else {
                content.style.display = 'none';
                icon.textContent = '▼';
            }
        });
    }

    // Modal overlay click to close
    const brainModal = document.getElementById('brain-modal');
    if (brainModal) {
        brainModal.querySelector('.modal-overlay').addEventListener('click', closeBrainModal);
    }
}

// ========================================
// INITIALIZATION
// ========================================

// Update main initialization to include Brain UI
const originalInit = window.onload;
window.onload = function() {
    if (originalInit) originalInit();

    // Initialize tab switching
    initTabSwitching();

    // Initialize Brain UI event handlers
    initBrainEventHandlers();

    console.log('🧠 Brain UI initialized');
};
