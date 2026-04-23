"""
bootstrap.py – run once from the project root:

    python source/bootstrap.py

Creates source/tests/ and writes the pytest suite for architecture_agent.
"""

import os
import textwrap

ROOT = os.path.dirname(__file__)   # …/source/
TESTS_DIR = os.path.join(ROOT, "tests")

# ---------------------------------------------------------------------------
# Test file content
# ---------------------------------------------------------------------------
TEST_CONTENT = textwrap.dedent('''\
    """
    Tests for architecture_agent package.

    Covers:
      - happy path  (read_spec, create_output_dirs, write_file, all generators)
      - missing spec (FileNotFoundError)
      - write failure (IOError via mocked open)
    """

    import os
    import sys
    import pytest
    from unittest.mock import patch, mock_open

    # Ensure the package on sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import architecture_agent as aa


    # ---------------------------------------------------------------------------
    # Fixtures
    # ---------------------------------------------------------------------------

    @pytest.fixture
    def tmp_spec(tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("# My Spec\\nSome requirements.", encoding="utf-8")
        return str(spec)


    @pytest.fixture
    def tmp_repo(tmp_path):
        return str(tmp_path / "repo")


    # ---------------------------------------------------------------------------
    # read_spec
    # ---------------------------------------------------------------------------

    class TestReadSpec:
        def test_happy_path(self, tmp_spec):
            content = aa.read_spec(tmp_spec)
            assert "My Spec" in content

        def test_missing_file_raises(self, tmp_path):
            with pytest.raises(FileNotFoundError, match="not found"):
                aa.read_spec(str(tmp_path / "nonexistent.md"))


    # ---------------------------------------------------------------------------
    # create_output_dirs
    # ---------------------------------------------------------------------------

    class TestCreateOutputDirs:
        def test_creates_diagrams_directory(self, tmp_repo):
            diagrams = aa.create_output_dirs(tmp_repo)
            assert os.path.isdir(diagrams)
            assert diagrams.endswith(os.path.join("docs", "architecture", "diagrams"))

        def test_idempotent(self, tmp_repo):
            aa.create_output_dirs(tmp_repo)
            # calling twice must not raise
            aa.create_output_dirs(tmp_repo)


    # ---------------------------------------------------------------------------
    # write_file
    # ---------------------------------------------------------------------------

    class TestWriteFile:
        def test_happy_path(self, tmp_path):
            target = str(tmp_path / "sub" / "out.md")
            result = aa.write_file(target, "hello")
            assert result["status"] == "success"
            assert result["path"] == target
            with open(target, encoding="utf-8") as fh:
                assert fh.read() == "hello"

        def test_write_failure_raises_ioerror(self, tmp_path):
            target = str(tmp_path / "fail.md")
            with patch("builtins.open", mock_open()) as m:
                m.side_effect = IOError("disk full")
                with pytest.raises(IOError, match="Failed to write file"):
                    aa.write_file(target, "content")


    # ---------------------------------------------------------------------------
    # Content generators
    # ---------------------------------------------------------------------------

    class TestGenerators:
        SPEC = "# Sample spec"

        def test_overview_md_is_nonempty(self):
            content = aa.generate_overview_md_content(self.SPEC)
            assert len(content) > 100
            assert "Architecture" in content

        def test_level1_context_is_valid_mermaid(self):
            content = aa.generate_c4_level1_context_mmd_content(self.SPEC)
            assert "flowchart" in content
            assert "User" in content

        def test_level2_container_is_valid_mermaid(self):
            content = aa.generate_c4_level2_container_mmd_content(self.SPEC)
            assert "flowchart" in content
            assert "subgraph" in content

        def test_level3_component_is_valid_mermaid(self):
            content = aa.generate_c4_level3_component_mmd_content(self.SPEC)
            assert "flowchart" in content
            assert "-->" in content

        def test_level4_code_is_classDiagram(self):
            content = aa.generate_c4_level4_code_mmd_content(self.SPEC)
            assert "classDiagram" in content
            assert "class " in content
''')


def main():
    os.makedirs(TESTS_DIR, exist_ok=True)

    init_path = os.path.join(TESTS_DIR, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as fh:
        fh.write("")   # empty marker

    test_path = os.path.join(TESTS_DIR, "test_architecture_agent.py")
    with open(test_path, "w", encoding="utf-8") as fh:
        fh.write(TEST_CONTENT)

    print(f"Created: {init_path}")
    print(f"Created: {test_path}")
    print("\nAll done. Run tests with:")
    print("  cd source && python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
