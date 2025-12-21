"""
Tests for vLLM systemd service template.

This test suite validates the systemd service template structure
and ensures it follows systemd best practices.
"""

import os
import pytest
import re


@pytest.fixture
def template_file_path():
    """Fixture providing the path to the service template file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "templates", "vllm.service.j2")


@pytest.fixture
def template_content(template_file_path):
    """Fixture to load the service template content."""
    with open(template_file_path, "r") as f:
        return f.read()


class TestVLLMServiceTemplate:
    """Test suite for vLLM systemd service template."""

    def test_template_file_exists(self, template_file_path):
        """Test that the template file exists."""
        assert os.path.exists(template_file_path), "vllm.service.j2 template should exist"

    def test_template_has_unit_section(self, template_content):
        """Test that template contains [Unit] section."""
        assert "[Unit]" in template_content, "Service should have [Unit] section"

    def test_template_has_service_section(self, template_content):
        """Test that template contains [Service] section."""
        assert "[Service]" in template_content, "Service should have [Service] section"

    def test_template_has_install_section(self, template_content):
        """Test that template contains [Install] section."""
        assert "[Install]" in template_content, "Service should have [Install] section"

    def test_unit_description_exists(self, template_content):
        """Test that service has a description."""
        assert re.search(r"Description\s*=\s*.+", template_content), \
            "Service should have a Description"
        description_match = re.search(r"Description\s*=\s*(.+)", template_content)
        if description_match:
            description = description_match.group(1).strip()
            assert "vllm" in description.lower(), "Description should mention vLLM"

    def test_unit_dependencies(self, template_content):
        """Test that service has proper dependencies."""
        # Should wait for network
        assert re.search(r"(Wants|After)\s*=.*network", template_content), \
            "Service should depend on network"
        # Should require docker
        assert re.search(r"(Requires|After)\s*=.*docker", template_content), \
            "Service should depend on Docker"

    def test_service_type_defined(self, template_content):
        """Test that service type is defined."""
        assert re.search(r"Type\s*=\s*\w+", template_content), \
            "Service should specify Type"

    def test_service_timeouts_defined(self, template_content):
        """Test that service has reasonable timeouts."""
        # Should have start timeout for model loading
        assert re.search(r"TimeoutStartSec\s*=\s*\d+", template_content), \
            "Service should specify TimeoutStartSec for model loading"

        # Should have stop timeout
        assert re.search(r"TimeoutStopSec\s*=\s*\d+", template_content), \
            "Service should specify TimeoutStopSec"

        # Extract and validate timeout values
        start_timeout_match = re.search(r"TimeoutStartSec\s*=\s*(\d+)", template_content)
        if start_timeout_match:
            start_timeout = int(start_timeout_match.group(1))
            assert start_timeout >= 300, \
                "TimeoutStartSec should be at least 300s for model loading"
            assert start_timeout <= 3600, \
                "TimeoutStartSec should not exceed 1 hour"

        stop_timeout_match = re.search(r"TimeoutStopSec\s*=\s*(\d+)", template_content)
        if stop_timeout_match:
            stop_timeout = int(stop_timeout_match.group(1))
            assert stop_timeout >= 30, "TimeoutStopSec should be at least 30s"
            assert stop_timeout <= 600, "TimeoutStopSec should not exceed 10 minutes"

    def test_exec_start_pre_commands(self, template_content):
        """Test that pre-start commands are defined."""
        assert "ExecStartPre" in template_content, \
            "Service should have ExecStartPre commands"

        # Should create volume
        assert re.search(r"ExecStartPre\s*=.*docker volume create", template_content), \
            "Should create Docker volume before starting"

        # Should remove old container
        assert re.search(r"ExecStartPre\s*=.*docker rm", template_content), \
            "Should remove old container before starting"
        # Check for --force flag
        assert re.search(r"docker rm.*--force", template_content), \
            "Should use --force flag when removing container"

    def test_exec_start_command(self, template_content):
        """Test that ExecStart uses docker run."""
        assert "ExecStart" in template_content, "Service should have ExecStart command"
        assert re.search(r"ExecStart\s*=.*docker run", template_content), \
            "ExecStart should use 'docker run'"

    def test_docker_run_essential_flags(self, template_content):
        """Test that docker run has essential flags."""
        # Should name the container
        assert re.search(r"--name\s*=\s*{{.*vllm_container_name", template_content), \
            "Docker run should name the container"

        # Should publish port
        assert re.search(r"--publish\s*=\s*{{.*vllm_port", template_content), \
            "Docker run should publish the port"

        # Should mount volume
        assert re.search(r"--volume\s*=\s*{{.*vllm_volume", template_content), \
            "Docker run should mount volume"

        # Should use GPUs
        assert re.search(r"--gpus", template_content), \
            "Docker run should configure GPUs"

        # Should use host IPC for better performance
        assert re.search(r"--ipc\s*=\s*host", template_content), \
            "Docker run should use host IPC namespace"

        # Should run in foreground (--detach=false or no --detach)
        if "--detach" in template_content:
            assert "--detach=false" in template_content, \
                "Docker should run in foreground for systemd Type=simple"

    def test_vllm_model_parameter(self, template_content):
        """Test that vLLM model parameter is specified."""
        # Should use --model parameter
        assert re.search(r"--model\s*=\s*{{.*vllm_model", template_content), \
            "vLLM should be configured with model parameter"

    def test_vllm_model_uses_variable(self, template_content):
        """
        Test that model parameter uses variable (validates 'reuse model name' intent).
        """
        model_matches = re.findall(r"--model\s*=\s*{{.*?}}", template_content)
        assert len(model_matches) >= 1, "Should have at least one --model parameter"

        for match in model_matches:
            assert "vllm_model" in match, \
                "Model should use vllm_model variable (reuse pattern)"
            # Should not have hardcoded model name
            assert "Qwen" not in match or "{{" in match, \
                "Model name should be a variable, not hardcoded"

    def test_vllm_essential_parameters(self, template_content):
        """Test that vLLM has essential runtime parameters."""
        # Should configure max model length
        assert re.search(r"--max-model-len\s*=\s*{{.*vllm_max_model_len", template_content), \
            "vLLM should configure max-model-len"

        # Should configure GPU memory
        assert re.search(r"--gpu-memory-utilization\s*=\s*{{.*vllm_gpu_memory_utilization", template_content), \
            "vLLM should configure gpu-memory-utilization"

        # Should configure tensor parallelism
        assert re.search(r"--tensor-parallel-size\s*=\s*{{.*vllm_tensor_parallel_size", template_content), \
            "vLLM should configure tensor-parallel-size"

    def test_vllm_tool_calling_configuration(self, template_content):
        """Test that vLLM is configured for tool calling."""
        # Should enable auto tool choice
        assert re.search(r"--enable-auto-tool-choice", template_content), \
            "vLLM should enable auto tool choice"

        # Should configure tool call parser
        assert re.search(r"--tool-call-parser\s*=\s*{{.*vllm_tool_call_parser", template_content), \
            "vLLM should configure tool call parser"

    def test_huggingface_token_environment(self, template_content):
        """Test that Hugging Face token is passed as environment variable."""
        assert re.search(r"--env\s*=\s*HUGGING_FACE_HUB_TOKEN", template_content), \
            "Should set HUGGING_FACE_HUB_TOKEN environment variable"

    def test_exec_stop_command(self, template_content):
        """Test that ExecStop gracefully stops container."""
        assert "ExecStop" in template_content, "Service should have ExecStop command"
        assert re.search(r"ExecStop\s*=.*docker stop", template_content), \
            "ExecStop should use 'docker stop' for graceful shutdown"

    def test_restart_policy(self, template_content):
        """Test that service has restart policy."""
        assert re.search(r"Restart\s*=", template_content), \
            "Service should have Restart policy"

        restart_match = re.search(r"Restart\s*=\s*(\w+)", template_content)
        if restart_match:
            restart_policy = restart_match.group(1)
            assert restart_policy in ["always", "on-failure", "unless-stopped"], \
                f"Restart policy should be standard value, got: {restart_policy}"

        # Should have restart delay
        assert re.search(r"RestartSec\s*=\s*\d+", template_content), \
            "Service should specify RestartSec"

    def test_install_wanted_by(self, template_content):
        """Test that service is wanted by multi-user target."""
        assert re.search(r"WantedBy\s*=\s*multi-user\.target", template_content), \
            "Service should be wanted by multi-user.target"


class TestVLLMServiceTemplateSecurity:
    """Test security aspects of the service template."""

    def test_no_hardcoded_secrets(self, template_content):
        """Test that no secrets are hardcoded."""
        # Should not contain actual HF tokens
        assert not re.search(r"hf_[a-zA-Z0-9]{30,}", template_content), \
            "Should not contain hardcoded Hugging Face tokens"

        # Should use variables for sensitive data
        if "HUGGING_FACE_HUB_TOKEN" in template_content:
            assert "{{" in template_content, \
                "Sensitive data should use template variables"

    def test_docker_security_flags(self, template_content):
        """Test that docker run uses appropriate security settings."""
        # Should not use --privileged (security risk)
        assert "--privileged" not in template_content, \
            "Should not use --privileged flag"

        # GPU access should be specific, not privileged
        if "--gpus" in template_content:
            gpus_match = re.search(r"--gpus\s+(\S+)", template_content)
            if gpus_match:
                gpus_value = gpus_match.group(1)
                # Should be either 'all' or specific device
                assert "{{" in gpus_value or gpus_value in ["all", "'all'", '"all"'], \
                    "GPUs should be configured via variable or 'all'"


class TestVLLMServiceTemplatePerformance:
    """Test performance-related configurations."""

    def test_ipc_host_for_shared_memory(self, template_content):
        """Test that IPC host mode is enabled for performance."""
        assert re.search(r"--ipc\s*=\s*host", template_content), \
            "Should use --ipc=host for shared memory performance"

    def test_volume_mount_for_cache(self, template_content):
        """Test that volume is mounted for model caching."""
        # Should mount HuggingFace cache directory
        assert re.search(r"\.cache/huggingface", template_content), \
            "Should mount HuggingFace cache directory"

        # Volume should use variable
        assert "vllm_volume" in template_content, \
            "Volume configuration should use variables"


class TestVLLMServiceTemplateReusability:
    """Test that template follows reusability best practices."""

    def test_all_values_are_variables(self, template_content):
        """Test that all configurable values use variables."""
        # Extract all variable usages
        variables = re.findall(r"{{.*?}}", template_content)
        assert len(variables) > 0, "Template should use variables"

        # Common vllm_ prefixed variables should be present
        template_lower = template_content.lower()
        expected_vars = [
            "vllm_container_name",
            "vllm_port",
            "vllm_volume_name",
            "vllm_volume_mount",
            "vllm_docker_image",
            "vllm_model",
            "vllm_gpu_devices",
        ]

        for var in expected_vars:
            assert var in template_lower, \
                f"Template should use variable: {var}"

    def test_no_hardcoded_paths(self, template_content):
        """Test that paths are not hardcoded."""
        # Docker image should use variable
        if "vllm/vllm-openai" in template_content:
            assert "{{" in template_content, \
                "Docker image should be configurable via variable"

    def test_jinja2_syntax_valid(self, template_content):
        """Test that Jinja2 syntax appears valid."""
        # Count opening and closing braces
        opening = template_content.count("{{")
        closing = template_content.count("}}")
        assert opening == closing, \
            "Jinja2 template should have matching braces"

        # No malformed variables
        assert not re.search(r"{{[^}]*$", template_content, re.MULTILINE), \
            "Should not have unclosed template variables"
        assert not re.search(r"^[^{]*}}", template_content, re.MULTILINE), \
            "Should not have unopened template variables"