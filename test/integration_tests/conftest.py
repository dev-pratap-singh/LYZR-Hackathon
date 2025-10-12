"""
Pytest configuration and fixtures for integration tests
"""
import pytest
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = str(Path(__file__).parent.parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Add app to path
app_path = str(Path(__file__).parent.parent.parent / "backend" / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory path"""
    return Path(__file__).parent.parent / "public"


@pytest.fixture(scope="session")
def test_csv_path(test_data_dir):
    """Get path to test CSV file with questions and ground truth"""
    return test_data_dir / "harrier_ev_detailed_qa.csv"


@pytest.fixture(scope="session")
def test_pdf_path(test_data_dir):
    """Get path to test PDF document"""
    return test_data_dir / "harrier-ev-all-you-need-to-know.pdf"


# Dev Singh test data fixtures
@pytest.fixture(scope="session")
def dev_singh_csv_path(test_data_dir):
    """Get path to Dev Singh's Q&A CSV file"""
    return test_data_dir / "dev_singh_ai_engineer_qa.csv"


@pytest.fixture(scope="session")
def dev_singh_pdf_path(test_data_dir):
    """Get path to Dev Singh's resume PDF"""
    return test_data_dir / "Dev-Singh-AI-Engineer.pdf"
