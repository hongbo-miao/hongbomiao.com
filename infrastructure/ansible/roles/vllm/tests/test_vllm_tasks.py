"""
Tests for vLLM Ansible role tasks.

This test suite validates the Ansible tasks structure and ensures
they follow best practices and contain all necessary elements.
"""

import os
import pytest
import yaml


@pytest.fixture
def tasks_file_path():
    """Fixture providing the path to the tasks file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "tasks", "main.yaml")


@pytest.fixture
def tasks_config(tasks_file_path):
    """Fixture to load the tasks YAML configuration."""
    with open(tasks_file_path, "r") as f:
        return yaml.safe_load(f)


class TestVLLMTasks:
    """Test suite for vLLM Ansible tasks."""

    def test_tasks_file_exists(self, tasks_file_path):
        """Test that the tasks file exists."""
        assert os.path.exists(tasks_file_path), "tasks/main.yaml file should exist"

    def test_tasks_is_valid_yaml(self, tasks_file_path):
        """Test that tasks file contains valid YAML."""
        with open(tasks_file_path, "r") as f:
            config = yaml.safe_load(f)
        assert config is not None, "YAML should be valid and parseable"
        assert isinstance(config, list), "Tasks should be a list"

    def test_tasks_list_not_empty(self, tasks_config):
        """Test that tasks list is not empty."""
        assert len(tasks_config) > 0, "Tasks list should contain at least one task"

    def test_all_tasks_have_names(self, tasks_config):
        """Test that all tasks have descriptive names."""
        for idx, task in enumerate(tasks_config):
            assert "name" in task, f"Task at index {idx} should have a 'name' field"
            assert isinstance(task["name"], str), f"Task name at index {idx} should be a string"
            assert len(task["name"]) > 0, f"Task name at index {idx} should not be empty"
            # Task names should be descriptive (at least 10 chars)
            assert len(task["name"]) >= 10, \
                f"Task name at index {idx} should be descriptive: '{task['name']}'"

    def test_pull_docker_image_task_exists(self, tasks_config):
        """Test that docker image pull task exists."""
        task_names = [task.get("name", "") for task in tasks_config]
        assert any("docker" in name.lower() and "image" in name.lower() for name in task_names), \
            "Should have a task to pull Docker image"

    def test_docker_image_task_configuration(self, tasks_config):
        """Test that docker image task is properly configured."""
        docker_tasks = [
            task for task in tasks_config
            if "docker" in task.get("name", "").lower() and "image" in task.get("name", "").lower()
        ]
        assert len(docker_tasks) > 0, "Should have at least one docker image task"

        docker_task = docker_tasks[0]
        # Should use community.docker.docker_image module
        assert "community.docker.docker_image" in docker_task, \
            "Should use community.docker.docker_image module"

        docker_config = docker_task["community.docker.docker_image"]
        assert "name" in docker_config, "Docker image task should specify image name"
        assert "source" in docker_config, "Docker image task should specify source"
        assert docker_config["source"] == "pull", "Docker image source should be 'pull'"

        # Should register result for conditional restart
        assert "register" in docker_task, "Docker image task should register result"

    def test_systemd_service_task_exists(self, tasks_config):
        """Test that systemd service creation task exists."""
        task_names = [task.get("name", "") for task in tasks_config]
        assert any("systemd" in name.lower() and "service" in name.lower() for name in task_names), \
            "Should have a task to create systemd service"

    def test_systemd_template_task_configuration(self, tasks_config):
        """Test that systemd template task is properly configured."""
        template_tasks = [
            task for task in tasks_config
            if "template" in task.get("ansible.builtin.template", {}) or
               "ansible.builtin.template" in task
        ]

        if len(template_tasks) > 0:
            template_task = template_tasks[0]
            if "ansible.builtin.template" in template_task:
                template_config = template_task["ansible.builtin.template"]
                assert "src" in template_config, "Template task should specify source"
                assert "dest" in template_config, "Template task should specify destination"
                assert template_config["dest"].startswith("/etc/systemd/system/"), \
                    "Systemd service should be in /etc/systemd/system/"
                assert "mode" in template_config, "Template task should specify file mode"
                # Should register result for conditional restart
                assert "register" in template_task, "Template task should register result"

    def test_service_management_task_exists(self, tasks_config):
        """Test that service management task exists."""
        task_names = [task.get("name", "") for task in tasks_config]
        assert any("service" in name.lower() and ("running" in name.lower() or "enabled" in name.lower())
                   for name in task_names), \
            "Should have a task to manage service state"

    def test_service_management_task_configuration(self, tasks_config):
        """Test that service management task is properly configured."""
        service_tasks = [
            task for task in tasks_config
            if "ansible.builtin.systemd_service" in task or "ansible.builtin.systemd" in task
        ]

        assert len(service_tasks) > 0, "Should have a systemd service management task"

        service_task = service_tasks[0]
        module_key = "ansible.builtin.systemd_service" if "ansible.builtin.systemd_service" in service_task else "ansible.builtin.systemd"
        service_config = service_task[module_key]

        assert "name" in service_config, "Service task should specify service name"
        assert "state" in service_config, "Service task should specify desired state"
        assert "enabled" in service_config, "Service task should specify if enabled on boot"

        # Should enable service for automatic start
        assert service_config["enabled"] is True, "Service should be enabled on boot"

    def test_tasks_have_proper_ordering(self, tasks_config):
        """Test that tasks are in logical order."""
        task_names = [task.get("name", "").lower() for task in tasks_config]

        # Find indices of key tasks
        pull_idx = next((i for i, name in enumerate(task_names) if "pull" in name or "image" in name), -1)
        template_idx = next((i for i, name in enumerate(task_names) if "template" in name or "create" in name), -1)
        service_idx = next((i for i, name in enumerate(task_names) if "running" in name or "enabled" in name), -1)

        # Logical order: pull image -> create config -> start service
        if pull_idx >= 0 and template_idx >= 0:
            assert pull_idx < template_idx, \
                "Docker image should be pulled before creating service file"

        if template_idx >= 0 and service_idx >= 0:
            assert template_idx < service_idx, \
                "Service file should be created before starting service"

    def test_tasks_use_variables_not_hardcoded_values(self, tasks_config):
        """Test that tasks use variables instead of hardcoded values."""
        tasks_str = yaml.dump(tasks_config)
        # Should use Jinja2 template variables
        assert "{{" in tasks_str, "Tasks should use template variables"
        assert "vllm_" in tasks_str, "Tasks should reference vllm_ prefixed variables"


class TestVLLMTasksIdempotency:
    """Test idempotency and safety aspects of tasks."""

    def test_docker_image_task_idempotent(self, tasks_config):
        """Test that docker image pull is idempotent."""
        docker_tasks = [
            task for task in tasks_config
            if "community.docker.docker_image" in task
        ]

        if len(docker_tasks) > 0:
            docker_config = docker_tasks[0]["community.docker.docker_image"]
            # force_source should be false for idempotency
            if "force_source" in docker_config:
                assert docker_config["force_source"] is False, \
                    "force_source should be false for idempotency"

    def test_service_restart_logic(self, tasks_config):
        """Test that service restarts only when necessary."""
        service_tasks = [
            task for task in tasks_config
            if "ansible.builtin.systemd_service" in task or "ansible.builtin.systemd" in task
        ]

        if len(service_tasks) > 0:
            service_task = service_tasks[0]
            module_key = "ansible.builtin.systemd_service" if "ansible.builtin.systemd_service" in service_task else "ansible.builtin.systemd"
            service_config = service_task[module_key]

            # State should be conditional - restart if changed, otherwise start
            state = service_config.get("state", "")
            assert "{{" in str(state), \
                "Service state should be conditional based on changes"


class TestVLLMTasksBestPractices:
    """Test that tasks follow Ansible best practices."""

    def test_tasks_use_fqcn(self, tasks_config):
        """Test that tasks use Fully Qualified Collection Names."""
        for task in tasks_config:
            # Get module names (exclude special keys like 'name', 'register', etc.)
            module_keys = [k for k in task.keys()
                          if k not in ["name", "register", "vars", "when", "tags", "become"]]

            for module in module_keys:
                # FQCN format: namespace.collection.module
                if module.count(".") > 0:  # Has namespace
                    parts = module.split(".")
                    assert len(parts) >= 3, \
                        f"Module '{module}' should use FQCN format (namespace.collection.module)"

    def test_no_shell_or_command_modules(self, tasks_config):
        """Test that tasks avoid shell/command modules when possible."""
        for task in tasks_config:
            task_str = str(task)
            # Prefer specific modules over shell/command
            assert "ansible.builtin.shell" not in task_str, \
                "Avoid shell module; use specific modules instead"
            assert "ansible.builtin.command" not in task_str, \
                "Avoid command module; use specific modules instead"

    def test_tasks_handle_errors(self, tasks_config):
        """Test that tasks have appropriate error handling."""
        # Critical tasks should either register results or have error handling
        for task in tasks_config:
            task_name = task.get("name", "").lower()
            # Service management tasks should register or have retry logic
            if "service" in task_name or "docker" in task_name:
                has_register = "register" in task
                has_failed_when = "failed_when" in task
                has_ignore_errors = "ignore_errors" in task
                # At least one error handling mechanism
                assert has_register or has_failed_when or has_ignore_errors, \
                    f"Critical task '{task.get('name')}' should have error handling"