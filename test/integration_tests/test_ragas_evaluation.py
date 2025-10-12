"""
Integration test for RAGAS evaluation of RAG system
Tests the entire pipeline: document upload -> processing -> query -> evaluation
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict
import uuid

# Add backend to path
backend_path = str(Path(__file__).parent.parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.services.ragas_evaluator import RAGASEvaluator
from app.services.search_agent import SearchAgent
from app.utils.document_helpers import process_document_pipeline
from app.database import PostgresSessionLocal
from app.models import Document
from app.config import settings


class TestRAGASEvaluation:
    """Integration tests for RAGAS evaluation"""

    @pytest.mark.asyncio
    async def test_ragas_evaluation_with_test_data(self, test_csv_path, test_pdf_path):
        """
        Test RAGAS evaluation using test PDF and gold-standard CSV questions
        This is the main integration test that will run in CI/CD
        """
        print("\n" + "="*80)
        print("Starting RAGAS Integration Test")
        print("="*80)

        # Verify test files exist
        assert test_csv_path.exists(), f"Test CSV not found: {test_csv_path}"
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"

        print(f"\n✓ Test files found:")
        print(f"  - CSV: {test_csv_path}")
        print(f"  - PDF: {test_pdf_path}")

        # Initialize services
        ragas_evaluator = RAGASEvaluator()
        search_agent = SearchAgent()

        # Load test questions from CSV
        print(f"\n{'='*80}")
        print("Step 1: Loading test questions from CSV")
        print(f"{'='*80}")
        df_test = ragas_evaluator.load_test_data_from_csv(str(test_csv_path))
        questions = df_test["Question"].tolist()
        ground_truths = df_test["Answer"].tolist()

        print(f"✓ Loaded {len(questions)} test questions")
        print(f"\nSample questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"  {i}. {q[:100]}...")

        # Process the test PDF document
        print(f"\n{'='*80}")
        print("Step 2: Processing test PDF document")
        print(f"{'='*80}")

        # Create a temporary document record
        db_session = PostgresSessionLocal()
        try:
            print(f"Processing document: {test_pdf_path.name}")

            # Create document record
            document = Document(
                filename=test_pdf_path.name,
                filepath=str(test_pdf_path),
                original_filename=test_pdf_path.name,
                file_size=test_pdf_path.stat().st_size,
                file_type="pdf",
                processing_status="pending"
            )

            db_session.add(document)
            db_session.commit()
            db_session.refresh(document)

            document_id = str(document.id)
            print(f"Document ID: {document_id}")

            # Process the document using the pipeline
            await process_document_pipeline(document, db_session, method="pymupdf")

            print(f"✓ Document processed successfully")
            print(f"  - Document ID: {document_id}")
            print(f"  - Chunks created: {document.total_chunks}")
            print(f"  - Graph processed: {document.graph_processed}")

            # Ensure graph processing is done
            print("\nEnsuring graph processing...")
            await search_agent.ensure_graph_processed(db_session)
            print("✓ Graph processing complete")

        except Exception as e:
            print(f"✗ Error processing document: {e}")
            db_session.rollback()
            raise
        finally:
            # Don't close session yet, we need it for queries
            pass

        # Query the RAG system for each question
        print(f"\n{'='*80}")
        print("Step 3: Querying RAG system for each test question")
        print(f"{'='*80}")

        answers = []
        contexts = []

        for i, question in enumerate(questions, 1):
            print(f"\nQuery {i}/{len(questions)}: {question[:80]}...")

            try:
                # Use simple_query for non-streaming response
                response = await search_agent.simple_query(question, document_id)

                answer = response.get("answer", "")
                answers.append(answer)

                # Extract contexts from intermediate steps
                intermediate_steps = response.get("intermediate_steps", [])
                question_contexts = []

                for step in intermediate_steps:
                    if len(step) >= 2:
                        tool_output = step[1]
                        if isinstance(tool_output, str) and tool_output.strip():
                            # Extract text from tool output
                            question_contexts.append(tool_output)

                # If no contexts found, add empty list
                if not question_contexts:
                    question_contexts = ["No context retrieved"]

                contexts.append(question_contexts)

                print(f"  ✓ Answer length: {len(answer)} chars")
                print(f"  ✓ Contexts retrieved: {len(question_contexts)}")

            except Exception as e:
                print(f"  ✗ Error querying: {e}")
                answers.append(f"Error: {str(e)}")
                contexts.append(["Error retrieving context"])

        # Close database session
        db_session.close()

        # Run RAGAS evaluation
        print(f"\n{'='*80}")
        print("Step 4: Running RAGAS evaluation")
        print(f"{'='*80}")

        evaluation_results = ragas_evaluator.evaluate_rag_performance(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )

        # Display results
        summary = evaluation_results["summary"]

        print(f"\n{'='*80}")
        print("RAGAS EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"\nTotal Questions Evaluated: {summary['total_questions']}")
        print(f"\n{'Metric':<30} {'Score':<10}")
        print("-" * 40)
        print(f"{'Context Precision':<30} {summary['context_precision_mean']:.4f}")
        print(f"{'Context Recall':<30} {summary['context_recall_mean']:.4f}")
        print(f"{'Retriever F1 Score':<30} {summary['retriever_f1_score_mean']:.4f}")
        print(f"{'Answer Correctness':<30} {summary['answer_correctness_mean']:.4f}")
        print(f"{'Factual Correctness':<30} {summary['factual_correctness_mean']:.4f}")
        print("-" * 40)

        # Calculate overall score
        overall_score = (
            summary['context_precision_mean'] +
            summary['context_recall_mean'] +
            summary['answer_correctness_mean'] +
            summary['factual_correctness_mean']
        ) / 4

        print(f"\n{'Overall Average Score':<30} {overall_score:.4f}")
        print(f"{'='*80}\n")

        # Save results to JSON and CSV files
        print(f"{'='*80}")
        print("Step 5: Saving results to files")
        print(f"{'='*80}")

        saved_paths = ragas_evaluator.save_results(
            evaluation_results=evaluation_results,
            output_dir="test/results",
            prefix="ragas"
        )

        print(f"\n✓ Results saved successfully!")
        print(f"  - JSON summary: {saved_paths['json_path']}")
        print(f"  - CSV details: {saved_paths['csv_path']}")
        print(f"  - Latest summary: {saved_paths['latest_json_path']}")
        print()

        # Assert that we have reasonable performance
        # These thresholds can be adjusted based on requirements
        assert summary['total_questions'] == len(questions), "Not all questions were evaluated"
        assert summary['context_precision_mean'] > 0, "Context precision should be positive"
        assert summary['answer_correctness_mean'] > 0, "Answer correctness should be positive"

        print("✓ RAGAS evaluation completed successfully!")

        return evaluation_results

    @pytest.mark.asyncio
    async def test_ragas_evaluation_dev_singh(self, dev_singh_csv_path, dev_singh_pdf_path):
        """
        Test RAGAS evaluation using Dev Singh's resume and Q&A
        Tests the RAG system with a different document type (resume)
        """
        print("\n" + "="*80)
        print("Starting RAGAS Integration Test - Dev Singh Resume")
        print("="*80)

        # Verify test files exist
        assert dev_singh_csv_path.exists(), f"Test CSV not found: {dev_singh_csv_path}"
        assert dev_singh_pdf_path.exists(), f"Test PDF not found: {dev_singh_pdf_path}"

        print(f"\n✓ Test files found:")
        print(f"  - CSV: {dev_singh_csv_path}")
        print(f"  - PDF: {dev_singh_pdf_path}")

        # Initialize services
        ragas_evaluator = RAGASEvaluator()
        search_agent = SearchAgent()

        # Load test questions from CSV
        print(f"\n{'='*80}")
        print("Step 1: Loading test questions from CSV")
        print(f"{'='*80}")
        df_test = ragas_evaluator.load_test_data_from_csv(str(dev_singh_csv_path))
        questions = df_test["Question"].tolist()
        ground_truths = df_test["Answer"].tolist()

        print(f"✓ Loaded {len(questions)} test questions")
        print(f"\nSample questions:")
        for i, q in enumerate(questions[:3], 1):
            print(f"  {i}. {q[:100]}...")

        # Process the test PDF document
        print(f"\n{'='*80}")
        print("Step 2: Processing test PDF document")
        print(f"{'='*80}")

        # Create a temporary document record
        db_session = PostgresSessionLocal()
        try:
            print(f"Processing document: {dev_singh_pdf_path.name}")

            # Create document record
            document = Document(
                filename=dev_singh_pdf_path.name,
                filepath=str(dev_singh_pdf_path),
                original_filename=dev_singh_pdf_path.name,
                file_size=dev_singh_pdf_path.stat().st_size,
                file_type="pdf",
                processing_status="pending"
            )

            db_session.add(document)
            db_session.commit()
            db_session.refresh(document)

            document_id = str(document.id)
            print(f"Document ID: {document_id}")

            # Process the document using the pipeline
            await process_document_pipeline(document, db_session, method="pymupdf")

            print(f"✓ Document processed successfully")
            print(f"  - Document ID: {document_id}")
            print(f"  - Chunks created: {document.total_chunks}")
            print(f"  - Graph processed: {document.graph_processed}")

            # Ensure graph processing is done
            print("\nEnsuring graph processing...")
            await search_agent.ensure_graph_processed(db_session)
            print("✓ Graph processing complete")

        except Exception as e:
            print(f"✗ Error processing document: {e}")
            db_session.rollback()
            raise
        finally:
            # Don't close session yet, we need it for queries
            pass

        # Query the RAG system for each question
        print(f"\n{'='*80}")
        print("Step 3: Querying RAG system for each test question")
        print(f"{'='*80}")

        answers = []
        contexts = []

        for i, question in enumerate(questions, 1):
            print(f"\nQuery {i}/{len(questions)}: {question[:80]}...")

            try:
                # Use simple_query for non-streaming response
                response = await search_agent.simple_query(question, document_id)

                answer = response.get("answer", "")
                answers.append(answer)

                # Extract contexts from intermediate steps
                intermediate_steps = response.get("intermediate_steps", [])
                question_contexts = []

                for step in intermediate_steps:
                    if len(step) >= 2:
                        tool_output = step[1]
                        if isinstance(tool_output, str) and tool_output.strip():
                            # Extract text from tool output
                            question_contexts.append(tool_output)

                # If no contexts found, add empty list
                if not question_contexts:
                    question_contexts = ["No context retrieved"]

                contexts.append(question_contexts)

                print(f"  ✓ Answer length: {len(answer)} chars")
                print(f"  ✓ Contexts retrieved: {len(question_contexts)}")

            except Exception as e:
                print(f"  ✗ Error querying: {e}")
                answers.append(f"Error: {str(e)}")
                contexts.append(["Error retrieving context"])

        # Close database session
        db_session.close()

        # Run RAGAS evaluation
        print(f"\n{'='*80}")
        print("Step 4: Running RAGAS evaluation")
        print(f"{'='*80}")

        evaluation_results = ragas_evaluator.evaluate_rag_performance(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )

        # Display results
        summary = evaluation_results["summary"]

        print(f"\n{'='*80}")
        print("RAGAS EVALUATION RESULTS - DEV SINGH RESUME")
        print(f"{'='*80}")
        print(f"\nTotal Questions Evaluated: {summary['total_questions']}")
        print(f"\n{'Metric':<30} {'Score':<10}")
        print("-" * 40)
        print(f"{'Context Precision':<30} {summary['context_precision_mean']:.4f}")
        print(f"{'Context Recall':<30} {summary['context_recall_mean']:.4f}")
        print(f"{'Retriever F1 Score':<30} {summary['retriever_f1_score_mean']:.4f}")
        print(f"{'Answer Correctness':<30} {summary['answer_correctness_mean']:.4f}")
        print(f"{'Factual Correctness':<30} {summary['factual_correctness_mean']:.4f}")
        print("-" * 40)

        # Calculate overall score
        overall_score = (
            summary['context_precision_mean'] +
            summary['context_recall_mean'] +
            summary['answer_correctness_mean'] +
            summary['factual_correctness_mean']
        ) / 4

        print(f"\n{'Overall Average Score':<30} {overall_score:.4f}")
        print(f"{'='*80}\n")

        # Save results to JSON and CSV files
        print(f"{'='*80}")
        print("Step 5: Saving results to files")
        print(f"{'='*80}")

        saved_paths = ragas_evaluator.save_results(
            evaluation_results=evaluation_results,
            output_dir="test/results",
            prefix="ragas_dev_singh"
        )

        print(f"\n✓ Results saved successfully!")
        print(f"  - JSON summary: {saved_paths['json_path']}")
        print(f"  - CSV details: {saved_paths['csv_path']}")
        print(f"  - Latest summary: {saved_paths['latest_json_path']}")
        print()

        # Assert that we have reasonable performance
        assert summary['total_questions'] == len(questions), "Not all questions were evaluated"
        assert summary['context_precision_mean'] > 0, "Context precision should be positive"
        assert summary['answer_correctness_mean'] > 0, "Answer correctness should be positive"

        print("✓ RAGAS evaluation completed successfully!")

        return evaluation_results


def main():
    """
    Main function to run RAGAS evaluation directly (for manual testing)
    """
    import pandas as pd

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    test_csv_path = project_root / "test" / "public" / "harrier_ev_detailed_qa.csv"
    test_pdf_path = project_root / "test" / "public" / "harrier-ev-all-you-need-to-know.pdf"

    # Create mock fixtures
    class MockFixture:
        def __init__(self, path):
            self.path = Path(path)

        def exists(self):
            return self.path.exists()

        def __str__(self):
            return str(self.path)

    # Run test
    test = TestRAGASEvaluation()
    asyncio.run(test.test_ragas_evaluation_with_test_data(
        MockFixture(test_csv_path),
        MockFixture(test_pdf_path)
    ))


if __name__ == "__main__":
    main()
