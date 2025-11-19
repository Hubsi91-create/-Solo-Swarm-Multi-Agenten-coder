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

    card.dataset.taskId = approval.task_id;
    card.querySelector('.approval-type').textContent = approval.task_type;
    card.querySelector('.approval-priority').textContent = 'HIGH';
    card.querySelector('.approval-title').textContent = approval.task_id;
    card.querySelector('.approval-description').textContent = approval.description;

    // Metadata
    if (approval.metadata) {
        const metadataText = Object.entries(approval.metadata)
            .map(([k, v]) => `${k}: ${v}`)
            .join(' | ');
        card.querySelector('.approval-metadata').textContent = metadataText;
    }

    // Button handlers
    card.querySelector('.btn-approve').addEventListener('click', () => {
        handleApproval(approval.task_id, true);
    });

    card.querySelector('.btn-reject').addEventListener('click', () => {
        handleApproval(approval.task_id, false);
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

    // Refresh data periodically
    setInterval(refreshStats, 30000); // Every 30 seconds
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
