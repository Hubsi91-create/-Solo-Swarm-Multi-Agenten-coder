"""
Test Suite for Dashboard Frontend Integration

Tests:
1. Frontend files exist and are accessible
2. HTML structure is valid
3. JavaScript configuration is correct
4. CSS is properly formatted
5. WebSocket message parsing works correctly
"""

import pytest
import os
import re
import json
from pathlib import Path


# Paths
FRONTEND_DIR = Path(__file__).parent.parent / "dashboard" / "frontend"
HTML_FILE = FRONTEND_DIR / "index.html"
JS_FILE = FRONTEND_DIR / "app.js"
CSS_FILE = FRONTEND_DIR / "style.css"


class TestFrontendFiles:
    """Test that frontend files exist and are readable"""

    def test_frontend_directory_exists(self):
        """Test that frontend directory exists"""
        assert FRONTEND_DIR.exists(), f"Frontend directory not found: {FRONTEND_DIR}"
        assert FRONTEND_DIR.is_dir(), f"Frontend path is not a directory: {FRONTEND_DIR}"

    def test_html_file_exists(self):
        """Test that index.html exists"""
        assert HTML_FILE.exists(), f"HTML file not found: {HTML_FILE}"
        assert HTML_FILE.is_file(), f"HTML path is not a file: {HTML_FILE}"

    def test_js_file_exists(self):
        """Test that app.js exists"""
        assert JS_FILE.exists(), f"JavaScript file not found: {JS_FILE}"
        assert JS_FILE.is_file(), f"JavaScript path is not a file: {JS_FILE}"

    def test_css_file_exists(self):
        """Test that style.css exists"""
        assert CSS_FILE.exists(), f"CSS file not found: {CSS_FILE}"
        assert CSS_FILE.is_file(), f"CSS path is not a file: {CSS_FILE}"

    def test_files_are_readable(self):
        """Test that all files can be read"""
        html_content = HTML_FILE.read_text()
        assert len(html_content) > 0, "HTML file is empty"

        js_content = JS_FILE.read_text()
        assert len(js_content) > 0, "JavaScript file is empty"

        css_content = CSS_FILE.read_text()
        assert len(css_content) > 0, "CSS file is empty"


class TestHTMLStructure:
    """Test HTML structure and content"""

    @pytest.fixture(scope="class")
    def html_content(self):
        """Load HTML content"""
        return HTML_FILE.read_text()

    def test_html_doctype(self, html_content):
        """Test that HTML has proper DOCTYPE"""
        assert html_content.strip().startswith("<!DOCTYPE html>"), "HTML missing DOCTYPE declaration"

    def test_html_lang_attribute(self, html_content):
        """Test that html tag has lang attribute"""
        assert 'lang="en"' in html_content, "HTML missing lang attribute"

    def test_html_has_head(self, html_content):
        """Test that HTML has head section"""
        assert "<head>" in html_content, "HTML missing <head> tag"
        assert "</head>" in html_content, "HTML missing </head> tag"

    def test_html_has_body(self, html_content):
        """Test that HTML has body section"""
        assert "<body>" in html_content, "HTML missing <body> tag"
        assert "</body>" in html_content, "HTML missing </body> tag"

    def test_html_has_title(self, html_content):
        """Test that HTML has title"""
        assert "<title>" in html_content, "HTML missing <title> tag"
        assert "Solo-Swarm" in html_content, "HTML title missing Solo-Swarm"

    def test_html_links_css(self, html_content):
        """Test that HTML links to CSS file"""
        assert 'href="style.css"' in html_content, "HTML missing link to style.css"

    def test_html_links_js(self, html_content):
        """Test that HTML links to JavaScript file"""
        assert 'src="app.js"' in html_content, "HTML missing link to app.js"

    def test_html_has_dashboard_container(self, html_content):
        """Test that HTML has main dashboard container"""
        assert 'class="dashboard"' in html_content, "HTML missing dashboard container"

    def test_html_has_header(self, html_content):
        """Test that HTML has header section"""
        assert '<header' in html_content, "HTML missing header tag"

    def test_html_has_agent_grid(self, html_content):
        """Test that HTML has agent grid container"""
        assert 'id="agent-grid"' in html_content, "HTML missing agent grid"

    def test_html_has_approval_queue(self, html_content):
        """Test that HTML has approval queue"""
        assert 'id="approval-queue"' in html_content, "HTML missing approval queue"

    def test_html_has_activity_log(self, html_content):
        """Test that HTML has activity log"""
        assert 'id="activity-log"' in html_content, "HTML missing activity log"

    def test_html_has_cost_monitor(self, html_content):
        """Test that HTML has cost monitor"""
        assert 'id="cost-monitor"' in html_content, "HTML missing cost monitor"

    def test_html_has_approval_template(self, html_content):
        """Test that HTML has approval card template"""
        assert '<template id="approval-card-template"' in html_content, "HTML missing approval card template"

    def test_html_has_connection_status(self, html_content):
        """Test that HTML has connection status indicator"""
        assert 'id="connection-status"' in html_content, "HTML missing connection status"

    def test_html_has_stats_elements(self, html_content):
        """Test that HTML has stats elements"""
        stats_ids = [
            "stat-tasks",
            "stat-completed",
            "stat-in-progress",
            "stat-failed",
            "stat-validations",
            "stat-pass-rate"
        ]

        for stat_id in stats_ids:
            assert f'id="{stat_id}"' in html_content, f"HTML missing stat element: {stat_id}"


class TestJavaScriptStructure:
    """Test JavaScript code structure"""

    @pytest.fixture(scope="class")
    def js_content(self):
        """Load JavaScript content"""
        return JS_FILE.read_text()

    def test_js_has_config(self, js_content):
        """Test that JS has CONFIG object"""
        assert "const CONFIG" in js_content, "JavaScript missing CONFIG object"

    def test_js_has_websocket_url(self, js_content):
        """Test that JS has WebSocket URL configured"""
        assert "WS_URL" in js_content, "JavaScript missing WS_URL configuration"
        assert "ws://" in js_content or "wss://" in js_content, "JavaScript missing WebSocket protocol"

    def test_js_has_api_url(self, js_content):
        """Test that JS has API URL configured"""
        assert "API_URL" in js_content, "JavaScript missing API_URL configuration"

    def test_js_has_reconnect_logic(self, js_content):
        """Test that JS has reconnect logic"""
        assert "RECONNECT_DELAY" in js_content, "JavaScript missing RECONNECT_DELAY"
        assert "reconnect" in js_content.lower(), "JavaScript missing reconnect functionality"

    def test_js_has_state_object(self, js_content):
        """Test that JS has state management"""
        assert "const state" in js_content, "JavaScript missing state object"

    def test_js_has_websocket_handlers(self, js_content):
        """Test that JS has WebSocket event handlers"""
        handlers = [
            "handleWebSocketOpen",
            "handleWebSocketMessage",
            "handleWebSocketClose",
            "handleWebSocketError"
        ]

        for handler in handlers:
            assert handler in js_content, f"JavaScript missing handler: {handler}"

    def test_js_has_update_handlers(self, js_content):
        """Test that JS has update handlers for different message types"""
        handlers = [
            "handleAgentStatusUpdate",
            "handleTaskUpdate",
            "handleCostUpdate",
            "handleValidationResult",
            "handleApprovalRequest"
        ]

        for handler in handlers:
            assert handler in js_content, f"JavaScript missing update handler: {handler}"

    def test_js_has_agent_grid_logic(self, js_content):
        """Test that JS has agent grid functionality"""
        assert "initializeAgentGrid" in js_content, "JavaScript missing agent grid initialization"
        assert "updateAgentSlot" in js_content, "JavaScript missing agent slot update"

    def test_js_has_cost_monitor_logic(self, js_content):
        """Test that JS has cost monitor functionality"""
        assert "updateCostMonitor" in js_content, "JavaScript missing cost monitor update"
        assert "WARNING_THRESHOLD" in js_content, "JavaScript missing cost warning threshold"

    def test_js_has_approval_logic(self, js_content):
        """Test that JS has approval handling"""
        assert "handleApproval" in js_content, "JavaScript missing approval handler"
        assert "updateApprovalQueue" in js_content, "JavaScript missing approval queue update"

    def test_js_has_logging(self, js_content):
        """Test that JS has activity logging"""
        assert "addLog" in js_content, "JavaScript missing logging function"

    def test_js_has_dom_ready(self, js_content):
        """Test that JS waits for DOM ready"""
        assert "DOMContentLoaded" in js_content, "JavaScript missing DOMContentLoaded listener"


class TestCSSStructure:
    """Test CSS structure"""

    @pytest.fixture(scope="class")
    def css_content(self):
        """Load CSS content"""
        return CSS_FILE.read_text()

    def test_css_has_variables(self, css_content):
        """Test that CSS has custom properties"""
        assert ":root" in css_content, "CSS missing :root selector"
        assert "--" in css_content, "CSS missing custom properties"

    def test_css_has_dark_theme(self, css_content):
        """Test that CSS has dark theme colors"""
        assert "--bg-" in css_content, "CSS missing background color variables"

    def test_css_has_status_colors(self, css_content):
        """Test that CSS has status colors defined"""
        status_colors = ["idle", "busy", "thinking", "error"]

        for status in status_colors:
            assert f"--status-{status}" in css_content, f"CSS missing status color: {status}"

    def test_css_has_grid_layout(self, css_content):
        """Test that CSS uses grid layout"""
        assert "display: grid" in css_content or "display:grid" in css_content, "CSS missing grid layout"

    def test_css_has_animations(self, css_content):
        """Test that CSS has animations"""
        assert "@keyframes" in css_content, "CSS missing animations"

    def test_css_has_agent_slot_styles(self, css_content):
        """Test that CSS has agent slot styles"""
        assert ".agent-slot" in css_content, "CSS missing agent slot styles"

    def test_css_has_approval_card_styles(self, css_content):
        """Test that CSS has approval card styles"""
        assert ".approval-card" in css_content, "CSS missing approval card styles"

    def test_css_has_responsive_design(self, css_content):
        """Test that CSS has responsive design"""
        assert "@media" in css_content, "CSS missing media queries for responsive design"

    def test_css_has_scrollbar_styles(self, css_content):
        """Test that CSS has custom scrollbar styles"""
        assert "scrollbar" in css_content, "CSS missing scrollbar styles"

    def test_css_has_tooltip_styles(self, css_content):
        """Test that CSS has tooltip styles"""
        assert ".tooltip" in css_content, "CSS missing tooltip styles"


class TestWebSocketMessageHandling:
    """Test WebSocket message structure compatibility"""

    def test_agent_status_message_structure(self):
        """Test that agent status message format is correct"""
        # Simulate a message from the backend
        message = {
            "update_type": "agent_status",
            "data": {
                "agent_id": "agent_001",
                "status": "busy",
                "details": {
                    "current_task": "impl_001",
                    "agent_type": "coder_agent"
                }
            },
            "timestamp": "2025-01-01T00:00:00",
            "priority": 1
        }

        # Test that message can be serialized
        json_str = json.dumps(message)
        assert len(json_str) > 0

        # Test that message can be deserialized
        parsed = json.loads(json_str)
        assert parsed["update_type"] == "agent_status"
        assert "data" in parsed

    def test_cost_update_message_structure(self):
        """Test that cost update message format is correct"""
        message = {
            "update_type": "cost_update",
            "data": {
                "total_cost_usd": 5.23,
                "total_tokens": 125000,
                "breakdown": {
                    "sonnet": 4.50,
                    "haiku": 0.73
                }
            },
            "timestamp": "2025-01-01T00:00:00",
            "priority": 1
        }

        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["update_type"] == "cost_update"
        assert parsed["data"]["total_cost_usd"] == 5.23
        assert isinstance(parsed["data"]["total_tokens"], int)

    def test_approval_request_message_structure(self):
        """Test that approval request message format is correct"""
        message = {
            "update_type": "approval_request",
            "data": {
                "task_id": "deploy_001",
                "task_type": "deployment",
                "description": "Deploy to production",
                "metadata": {
                    "risk": "high",
                    "estimated_cost": 0.50
                }
            },
            "timestamp": "2025-01-01T00:00:00",
            "priority": 2
        }

        json_str = json.dumps(message)
        parsed = json.loads(json_str)

        assert parsed["update_type"] == "approval_request"
        assert "task_id" in parsed["data"]
        assert "description" in parsed["data"]


class TestConfiguration:
    """Test configuration values"""

    @pytest.fixture(scope="class")
    def js_content(self):
        """Load JavaScript content"""
        return JS_FILE.read_text()

    def test_websocket_url_format(self, js_content):
        """Test that WebSocket URL has correct format"""
        # Extract WS_URL value
        match = re.search(r"WS_URL:\s*['\"]([^'\"]+)['\"]", js_content)
        assert match is not None, "Could not find WS_URL in JavaScript"

        ws_url = match.group(1)
        assert ws_url.startswith("ws://") or ws_url.startswith("wss://"), \
            f"Invalid WebSocket URL format: {ws_url}"

    def test_api_url_format(self, js_content):
        """Test that API URL has correct format"""
        # Extract API_URL value
        match = re.search(r"API_URL:\s*['\"]([^'\"]+)['\"]", js_content)
        assert match is not None, "Could not find API_URL in JavaScript"

        api_url = match.group(1)
        assert api_url.startswith("http://") or api_url.startswith("https://"), \
            f"Invalid API URL format: {api_url}"

    def test_reconnect_delay_is_reasonable(self, js_content):
        """Test that reconnect delay is reasonable"""
        # Extract RECONNECT_DELAY value
        match = re.search(r"RECONNECT_DELAY:\s*(\d+)", js_content)
        assert match is not None, "Could not find RECONNECT_DELAY in JavaScript"

        delay = int(match.group(1))
        assert 1000 <= delay <= 10000, f"Reconnect delay should be 1-10 seconds, got {delay}ms"

    def test_daily_budget_is_set(self, js_content):
        """Test that daily budget is configured"""
        # Extract DAILY_BUDGET value
        match = re.search(r"DAILY_BUDGET:\s*([\d.]+)", js_content)
        assert match is not None, "Could not find DAILY_BUDGET in JavaScript"

        budget = float(match.group(1))
        assert budget > 0, f"Daily budget should be positive, got {budget}"

    def test_warning_threshold_is_reasonable(self, js_content):
        """Test that warning threshold is between 0 and 1"""
        # Extract WARNING_THRESHOLD value
        match = re.search(r"WARNING_THRESHOLD:\s*([\d.]+)", js_content)
        assert match is not None, "Could not find WARNING_THRESHOLD in JavaScript"

        threshold = float(match.group(1))
        assert 0 < threshold < 1, f"Warning threshold should be 0-1, got {threshold}"


class TestUIElements:
    """Test that UI elements are properly referenced"""

    @pytest.fixture(scope="class")
    def html_content(self):
        """Load HTML content"""
        return HTML_FILE.read_text()

    @pytest.fixture(scope="class")
    def js_content(self):
        """Load JavaScript content"""
        return JS_FILE.read_text()

    def test_all_js_element_ids_exist_in_html(self, html_content, js_content):
        """Test that all element IDs referenced in JS exist in HTML"""
        # Extract all getElementById calls
        id_matches = re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", js_content)

        for element_id in set(id_matches):
            assert f'id="{element_id}"' in html_content, \
                f"Element ID '{element_id}' used in JS but not found in HTML"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
