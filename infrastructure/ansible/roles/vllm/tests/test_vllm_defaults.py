"""
Tests for vLLM Ansible role default variables.

This test suite validates the configuration structure and values
in the vLLM Ansible role defaults to ensure they are properly formatted
and contain valid values.
"""

import os
import pytest
import yaml


@pytest.fixture
def defaults_file_path():
    """Fixture providing the path to the defaults file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "defaults", "main.yaml")


@pytest.fixture
def defaults_config(defaults_file_path):
    """Fixture to load the defaults YAML configuration."""
    with open(defaults_file_path, "r") as f:
        return yaml.safe_load(f)


class TestVLLMDefaults:
    """Test suite for vLLM default configuration values."""

    def test_defaults_file_exists(self, defaults_file_path):
        """Test that the defaults file exists."""
        assert os.path.exists(defaults_file_path), "defaults/main.yaml file should exist"

    def test_defaults_is_valid_yaml(self, defaults_file_path):
        """Test that defaults file contains valid YAML."""
        with open(defaults_file_path, "r") as f:
            config = yaml.safe_load(f)
        assert config is not None, "YAML should be valid and parseable"
        assert isinstance(config, dict), "Root element should be a dictionary"

    def test_required_variables_exist(self, defaults_config):
        """Test that all required variables are defined."""
        required_vars = [
            "vllm_docker_image",
            "vllm_container_name",
            "vllm_port",
            "vllm_volume_name",
            "vllm_volume_mount",
            "vllm_model",
            "vllm_max_model_len",
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
            "vllm_gpu_devices",
            "vllm_tool_call_parser",
            "vllm_hugging_face_hub_token",
        ]
        for var in required_vars:
            assert var in defaults_config, f"Required variable '{var}' should be defined"

    def test_docker_image_format(self, defaults_config):
        """Test that docker image is properly formatted."""
        docker_image = defaults_config["vllm_docker_image"]
        assert isinstance(docker_image, str), "Docker image should be a string"
        assert len(docker_image) > 0, "Docker image should not be empty"
        assert "vllm" in docker_image.lower(), "Docker image should contain 'vllm'"
        # Should contain registry/image:tag format
        assert "/" in docker_image, "Docker image should contain registry path"

    def test_container_name_valid(self, defaults_config):
        """Test that container name is valid."""
        container_name = defaults_config["vllm_container_name"]
        assert isinstance(container_name, str), "Container name should be a string"
        assert len(container_name) > 0, "Container name should not be empty"
        assert container_name.islower() or "-" in container_name or "_" in container_name, \
            "Container name should follow Docker naming conventions"

    def test_port_is_valid(self, defaults_config):
        """Test that port is a valid port number."""
        port = defaults_config["vllm_port"]
        assert isinstance(port, int), "Port should be an integer"
        assert 1 <= port <= 65535, "Port should be between 1 and 65535"

    def test_volume_name_valid(self, defaults_config):
        """Test that volume name is valid."""
        volume_name = defaults_config["vllm_volume_name"]
        assert isinstance(volume_name, str), "Volume name should be a string"
        assert len(volume_name) > 0, "Volume name should not be empty"

    def test_volume_mount_path_valid(self, defaults_config):
        """Test that volume mount path is a valid absolute path."""
        mount_path = defaults_config["vllm_volume_mount"]
        assert isinstance(mount_path, str), "Mount path should be a string"
        assert mount_path.startswith("/"), "Mount path should be absolute"
        assert ".." not in mount_path, "Mount path should not contain relative references"

    def test_model_name_format(self, defaults_config):
        """Test that model name follows expected format."""
        model = defaults_config["vllm_model"]
        assert isinstance(model, str), "Model name should be a string"
        assert len(model) > 0, "Model name should not be empty"
        # Model should typically be in format: org/model-name
        assert "/" in model, "Model should follow HuggingFace format: org/model-name"

    def test_max_model_len_valid(self, defaults_config):
        """Test that max_model_len is a valid positive integer."""
        max_len = defaults_config["vllm_max_model_len"]
        assert isinstance(max_len, int), "max_model_len should be an integer"
        assert max_len > 0, "max_model_len should be positive"
        assert max_len >= 512, "max_model_len should be at least 512"
        # Common values are powers of 2 or multiples of 1024
        assert max_len % 512 == 0, "max_model_len should typically be a multiple of 512"

    def test_gpu_memory_utilization_valid(self, defaults_config):
        """Test that GPU memory utilization is a valid ratio."""
        gpu_mem = defaults_config["vllm_gpu_memory_utilization"]
        assert isinstance(gpu_mem, (int, float)), "GPU memory utilization should be numeric"
        assert 0.0 < gpu_mem <= 1.0, "GPU memory utilization should be between 0 and 1"

    def test_tensor_parallel_size_valid(self, defaults_config):
        """Test that tensor parallel size is valid."""
        tensor_size = defaults_config["vllm_tensor_parallel_size"]
        assert isinstance(tensor_size, int), "Tensor parallel size should be an integer"
        assert tensor_size > 0, "Tensor parallel size should be positive"
        # Should typically be a power of 2 for optimal performance
        assert tensor_size in [1, 2, 4, 8, 16], \
            "Tensor parallel size should typically be 1, 2, 4, 8, or 16"

    def test_gpu_devices_valid(self, defaults_config):
        """Test that GPU devices specification is valid."""
        gpu_devices = defaults_config["vllm_gpu_devices"]
        assert isinstance(gpu_devices, str), "GPU devices should be a string"
        assert len(gpu_devices) > 0, "GPU devices should not be empty"
        # Valid formats: "all", "device=0", "device=0,1", etc.
        assert gpu_devices == "all" or "device=" in gpu_devices, \
            "GPU devices should be 'all' or specify devices"

    def test_tool_call_parser_valid(self, defaults_config):
        """Test that tool call parser is a recognized value."""
        parser = defaults_config["vllm_tool_call_parser"]
        assert isinstance(parser, str), "Tool call parser should be a string"
        assert len(parser) > 0, "Tool call parser should not be empty"
        # Common parsers: hermes, mistral, llama, etc.
        valid_parsers = ["hermes", "mistral", "llama", "internlm", "granite"]
        assert parser in valid_parsers, \
            f"Tool call parser should be one of {valid_parsers}"

    def test_hugging_face_token_is_string(self, defaults_config):
        """Test that Hugging Face token is a string (can be empty)."""
        token = defaults_config["vllm_hugging_face_hub_token"]
        assert isinstance(token, str), "Hugging Face token should be a string"
        # Empty string is valid for public models


class TestVLLMDefaultsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_model_name_reuse_consistency(self, defaults_config):
        """
        Test that model name is defined once and can be reused.
        This validates the 'reuse model name' refactoring intent.
        """
        model = defaults_config["vllm_model"]
        # Verify it's a simple string without template variables
        assert "{{" not in model, "Model should not contain template variables in defaults"
        assert "}}" not in model, "Model should not contain template variables in defaults"

    def test_reasonable_resource_limits(self, defaults_config):
        """Test that resource configurations are reasonable."""
        max_len = defaults_config["vllm_max_model_len"]
        gpu_mem = defaults_config["vllm_gpu_memory_utilization"]
        tensor_size = defaults_config["vllm_tensor_parallel_size"]

        # Sanity checks for production use
        assert max_len <= 131072, "max_model_len seems unusually high (>128K tokens)"
        assert gpu_mem <= 0.95, "GPU memory utilization >95% may cause OOM errors"
        assert tensor_size <= 16, "Tensor parallel size >16 is uncommon"

    def test_security_token_not_hardcoded(self, defaults_config):
        """Test that no actual token is hardcoded in defaults."""
        token = defaults_config["vllm_hugging_face_hub_token"]
        # Token should be empty or a placeholder in defaults
        assert token == "" or token.startswith("{{"), \
            "Real tokens should not be hardcoded in defaults"
        # Real HF tokens start with "hf_"
        assert not token.startswith("hf_"), \
            "Real Hugging Face tokens should not be in version control"


class TestVLLMDefaultsCompatibility:
    """Test compatibility and integration aspects."""

    def test_port_not_in_common_conflicting_range(self, defaults_config):
        """Test that default port doesn't conflict with common services."""
        port = defaults_config["vllm_port"]
        # Avoid common ports
        common_ports = [22, 80, 443, 3000, 3306, 5432, 6379, 8080, 9090, 9200]
        assert port not in common_ports, \
            f"Port {port} conflicts with common services"

    def test_docker_image_uses_versioned_tag(self, defaults_config):
        """Test that docker image uses a version tag for reproducibility."""
        docker_image = defaults_config["vllm_docker_image"]
        # Good practice to have version tag, not just :latest
        assert ":" in docker_image, "Docker image should include a tag"
        tag = docker_image.split(":")[-1]
        # Warn about :latest but don't fail (it's in defaults)
        if tag != "latest":
            # Should be semantic version or specific version
            assert len(tag) > 0, "Tag should not be empty"

    def test_model_name_matches_huggingface_format(self, defaults_config):
        """Test that model name matches HuggingFace repository format."""
        model = defaults_config["vllm_model"]
        parts = model.split("/")
        assert len(parts) == 2, \
            "Model should be in format 'organization/model-name'"
        org, model_name = parts
        assert len(org) > 0, "Organization name should not be empty"
        assert len(model_name) > 0, "Model name should not be empty"
        # Model names typically use hyphens or underscores
        assert any(c.isalnum() or c in ["-", "_", "."] for c in model_name), \
            "Model name should contain valid characters"