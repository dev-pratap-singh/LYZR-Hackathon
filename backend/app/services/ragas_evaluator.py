"""
RAGAS Evaluator for RAG System Performance Testing
"""
import pandas as pd
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import (
    answer_correctness,
    context_recall,
    context_precision,
    FactualCorrectness
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluator for RAG system using RAGAS framework"""

    def __init__(self):
        """Initialize RAGAS Evaluator with LLM and embeddings"""
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self.evaluator_llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self.evaluator_embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )

        logger.info("RAGASEvaluator initialized")

    def evaluate_rag_performance(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict:
        """
        Evaluate RAG system performance using RAGAS metrics

        Parameters:
        -----------
        questions : List[str]
            List of user questions
        answers : List[str]
            List of system-generated answers
        contexts : List[List[str]]
            List of retrieved contexts for each question (list of context strings)
        ground_truths : List[str]
            List of expected/reference answers

        Returns:
        --------
        Dict containing:
            - results_df: DataFrame with per-question metrics
            - summary: Dict with average metrics
        """
        try:
            logger.info(f"Starting RAGAS evaluation for {len(questions)} questions")

            # Validate inputs
            if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
                raise ValueError(
                    f"Input length mismatch: questions={len(questions)}, "
                    f"answers={len(answers)}, contexts={len(contexts)}, "
                    f"ground_truths={len(ground_truths)}"
                )

            # Create dataset
            dataset = []
            for i in tqdm(range(len(questions)), desc="Processing evaluation data"):
                question = questions[i]
                answer = answers[i]
                context = contexts[i]
                reference = ground_truths[i]

                # Convert context to list if it's not already
                if isinstance(context, str):
                    context = [context]

                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=context,
                    reference=reference,
                )
                dataset.append(sample)

            # Configure metrics
            factual_correctness = FactualCorrectness(
                mode="precision",
                atomicity="high",
                coverage="high"
            )

            # Create evaluation dataset and run evaluation
            logger.info("Running RAGAS evaluation...")
            evaluation_data = EvaluationDataset(samples=dataset)
            result = evaluate(
                dataset=evaluation_data,
                metrics=[
                    context_recall,
                    context_precision,
                    answer_correctness,
                    factual_correctness,
                ],
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings,
            )

            # Convert to pandas DataFrame
            df_result = result.to_pandas()

            # Rename columns for consistency
            if "factual_correctness(mode=precision)" in df_result.columns:
                df_result.rename(
                    columns={"factual_correctness(mode=precision)": "factual_correctness"},
                    inplace=True
                )

            # Calculate F1 score for retriever performance
            df_result["retriever_f1_score"] = (
                2
                * df_result["context_recall"]
                * df_result["context_precision"]
                / (df_result["context_recall"] + df_result["context_precision"])
            ).fillna(0)  # Handle division by zero

            # Calculate summary statistics
            summary = {
                "context_precision_mean": float(df_result["context_precision"].mean()),
                "context_recall_mean": float(df_result["context_recall"].mean()),
                "retriever_f1_score_mean": float(df_result["retriever_f1_score"].mean()),
                "answer_correctness_mean": float(df_result["answer_correctness"].mean()),
                "factual_correctness_mean": float(df_result["factual_correctness"].mean()),
                "total_questions": len(questions),
            }

            logger.info(f"RAGAS evaluation completed. Summary: {summary}")

            return {
                "results_df": df_result,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}", exc_info=True)
            raise

    def load_test_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load test questions and ground truth answers from CSV file

        Parameters:
        -----------
        csv_path : str
            Path to CSV file with columns: Question, Answer

        Returns:
        --------
        pd.DataFrame with questions and ground truth answers
        """
        try:
            df = pd.read_csv(csv_path)

            # Validate required columns
            if "Question" not in df.columns or "Answer" not in df.columns:
                raise ValueError(
                    f"CSV must have 'Question' and 'Answer' columns. "
                    f"Found columns: {df.columns.tolist()}"
                )

            logger.info(f"Loaded {len(df)} test questions from {csv_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {e}", exc_info=True)
            raise

    def save_results(
        self,
        evaluation_results: Dict,
        output_dir: str = "test/results",
        prefix: str = "ragas"
    ) -> Dict[str, str]:
        """
        Save RAGAS evaluation results to JSON and CSV files with timestamps

        Parameters:
        -----------
        evaluation_results : Dict
            Results dictionary from evaluate_rag_performance()
        output_dir : str
            Directory to save results (default: test/results)
        prefix : str
            Prefix for output files (default: ragas)

        Returns:
        --------
        Dict with paths to saved files:
            - json_path: Path to JSON summary file
            - csv_path: Path to CSV detailed results file
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Prepare summary for JSON
            summary = evaluation_results["summary"].copy()
            summary["timestamp"] = datetime.now().isoformat()
            summary["test_date"] = datetime.now().strftime("%Y-%m-%d")
            summary["test_time"] = datetime.now().strftime("%H:%M:%S")

            # Save JSON summary
            json_filename = f"{prefix}_summary_{timestamp}.json"
            json_path = output_path / json_filename
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved JSON summary to: {json_path}")

            # Save detailed results to CSV
            csv_filename = f"{prefix}_detailed_{timestamp}.csv"
            csv_path = output_path / csv_filename
            results_df = evaluation_results["results_df"].copy()

            # Add metadata columns
            results_df.insert(0, "timestamp", datetime.now().isoformat())
            results_df.insert(1, "test_date", datetime.now().strftime("%Y-%m-%d"))

            results_df.to_csv(csv_path, index=False)
            logger.info(f"Saved detailed results to: {csv_path}")

            # Also save/update a latest summary file for easy access
            latest_json_path = output_path / f"{prefix}_latest_summary.json"
            with open(latest_json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Updated latest summary: {latest_json_path}")

            return {
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "latest_json_path": str(latest_json_path)
            }

        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)
            raise
