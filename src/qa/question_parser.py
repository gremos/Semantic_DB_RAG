"""
Question Parser - Phase 3
Parses natural language questions and generates SQL with evidence.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from config.settings import get_settings, get_path_config

logger = logging.getLogger(__name__)


class QuestionParser:
    """Parse natural language questions and generate SQL with execution."""

    def __init__(self, semantic_model: Optional[Dict[str, Any]] = None):
        """
        Initialize parser with semantic model

        Args:
            semantic_model: Pre-loaded semantic model (loads from cache if not provided)
        """
        self.settings = get_settings()
        self.path_config = get_path_config()

        # Load semantic model
        if semantic_model:
            self.semantic_model = semantic_model
        else:
            self.semantic_model = self._load_semantic_model()

        # Initialize components lazily
        self._sql_generator = None
        self._sql_executor = None
        self._history_file = self.path_config.cache_dir / 'qa_history.jsonl'

    def _load_semantic_model(self) -> Dict[str, Any]:
        """Load semantic model from cache"""
        cache_file = self.path_config.cache_dir / 'semantic_model.json'

        if not cache_file.exists():
            raise FileNotFoundError(
                "Semantic model not found. Run: python main.py model"
            )

        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @property
    def sql_generator(self):
        """Lazy-load SQL generator"""
        if self._sql_generator is None:
            from src.qa.sql_generator import SQLGenerator
            self._sql_generator = SQLGenerator(self.semantic_model)
        return self._sql_generator

    @property
    def sql_executor(self):
        """Lazy-load SQL executor"""
        if self._sql_executor is None:
            from src.qa.sql_executor import SQLExecutor
            self._sql_executor = SQLExecutor()
        return self._sql_executor

    def answer(
        self,
        question: str,
        execute: bool = True,
        row_limit: int = 100,
        timeout_sec: int = 60
    ) -> Dict[str, Any]:
        """
        Answer a natural language question with SQL.

        Steps:
        1. Generate SQL from question using semantic model
        2. Validate SQL (read-only check)
        3. Execute SQL if requested
        4. Format results
        5. Log to Q&A history

        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            row_limit: Maximum rows to return
            timeout_sec: Query timeout in seconds

        Returns:
            {
                'status': 'ok'|'refuse'|'error',
                'question': str,
                'sql': {...} generated SQL info,
                'results': {...} execution results if execute=True,
                'confidence': float,
                'evidence': {...},
                'refusal': {...} if status='refuse',
                'error': str if status='error'
            }
        """
        logger.info(f"Answering question: {question}")
        start_time = datetime.now()

        response = {
            'question': question,
            'timestamp': start_time.isoformat(),
            'status': 'ok'
        }

        try:
            # Step 1: Generate SQL
            logger.info("Step 1: Generating SQL...")
            sql_result = self.sql_generator.generate_sql(question, row_limit=row_limit)

            response['confidence'] = sql_result.get('confidence', 0.0)
            response['evidence'] = sql_result.get('evidence', {})

            # Step 2: Check if generation was refused
            if sql_result.get('status') == 'refuse':
                logger.info(f"SQL generation refused: {sql_result.get('refusal', {}).get('reason')}")
                response['status'] = 'refuse'
                response['refusal'] = sql_result.get('refusal', {})
                self._log_history(response)
                return response

            # Step 3: Extract SQL statement
            sql_info = sql_result.get('sql', [{}])[0]
            sql_statement = sql_info.get('statement', '')

            response['sql'] = {
                'statement': sql_statement,
                'dialect': sql_info.get('dialect', 'tsql'),
                'explanation': sql_info.get('explanation', ''),
                'evidence': sql_info.get('evidence', {})
            }

            # Step 4: Validate read-only
            if not self.sql_executor.validate_read_only(sql_statement):
                logger.error("Generated SQL is not read-only!")
                response['status'] = 'error'
                response['error'] = "Generated SQL contains write operations (blocked)"
                self._log_history(response)
                return response

            # Step 5: Execute if requested
            if execute and sql_statement:
                logger.info("Step 2: Executing SQL...")
                exec_result = self.sql_executor.execute(
                    sql_statement,
                    timeout_sec=timeout_sec,
                    row_limit=row_limit
                )

                response['results'] = {
                    'status': exec_result.get('status'),
                    'row_count': exec_result.get('row_count', 0),
                    'columns': exec_result.get('columns', []),
                    'data': exec_result.get('data', []),
                    'execution_time_ms': exec_result.get('execution_time_ms', 0),
                    'truncated': exec_result.get('truncated', False)
                }

                if exec_result.get('status') == 'error':
                    response['status'] = 'error'
                    response['error'] = exec_result.get('error', 'Execution failed')
                elif exec_result.get('status') == 'timeout':
                    response['status'] = 'error'
                    response['error'] = exec_result.get('error', 'Query timed out')

            # Step 6: Log to history
            response['total_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            self._log_history(response)

            logger.info(f"Answer complete: status={response['status']}, "
                       f"confidence={response.get('confidence', 0):.0%}, "
                       f"time={response['total_time_ms']:.0f}ms")

            return response

        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            response['status'] = 'error'
            response['error'] = str(e)
            self._log_history(response)
            return response

    def explain(self, question: str) -> Dict[str, Any]:
        """
        Explain how a question would be answered without executing

        Args:
            question: Natural language question

        Returns:
            Explanation with SQL and evidence but no execution
        """
        return self.answer(question, execute=False)

    def format_answer(self, response: Dict[str, Any], format_type: str = 'text') -> str:
        """
        Format answer response for display

        Args:
            response: Answer response dict
            format_type: 'text', 'json', or 'markdown'

        Returns:
            Formatted string
        """
        if format_type == 'json':
            return json.dumps(response, indent=2, default=str)

        lines = []

        # Question
        lines.append(f"Q: {response.get('question', '')}")
        lines.append("")

        # Status
        status = response.get('status', 'unknown')
        confidence = response.get('confidence', 0)

        if status == 'refuse':
            lines.append(f"Status: REFUSED (confidence: {confidence:.0%})")
            refusal = response.get('refusal', {})
            lines.append(f"Reason: {refusal.get('reason', 'Unknown')}")

            clarifications = refusal.get('clarifying_questions', [])
            if clarifications:
                lines.append("\nClarifying questions:")
                for q in clarifications:
                    lines.append(f"  - {q}")
            return '\n'.join(lines)

        if status == 'error':
            lines.append(f"Status: ERROR")
            lines.append(f"Error: {response.get('error', 'Unknown error')}")
            return '\n'.join(lines)

        # SQL
        lines.append(f"Status: OK (confidence: {confidence:.0%})")
        sql_info = response.get('sql', {})
        if sql_info:
            lines.append(f"\nSQL ({sql_info.get('dialect', 'tsql')}):")
            lines.append(sql_info.get('statement', ''))
            if sql_info.get('explanation'):
                lines.append(f"\nExplanation: {sql_info['explanation']}")

        # Results
        results = response.get('results', {})
        if results:
            lines.append(f"\nResults: {results.get('row_count', 0)} rows "
                        f"({results.get('execution_time_ms', 0):.0f}ms)")

            if results.get('data'):
                # Format as simple table
                lines.append(self.sql_executor.format_results(results, 'table'))

        # Evidence
        evidence = response.get('evidence', {})
        if evidence and format_type == 'markdown':
            lines.append("\n### Evidence")
            if evidence.get('entities'):
                lines.append(f"- Entities: {', '.join(evidence['entities'])}")
            if evidence.get('measures'):
                lines.append(f"- Measures: {', '.join(evidence['measures'])}")

        return '\n'.join(lines)

    def _log_history(self, response: Dict[str, Any]):
        """
        Log Q&A interaction to history file (JSONL format)

        Args:
            response: Full response dict to log
        """
        try:
            # Ensure cache dir exists
            self._history_file.parent.mkdir(parents=True, exist_ok=True)

            # Create log entry
            log_entry = {
                'timestamp': response.get('timestamp', datetime.now().isoformat()),
                'question': response.get('question', ''),
                'status': response.get('status', ''),
                'confidence': response.get('confidence', 0),
                'sql_statement': response.get('sql', {}).get('statement', ''),
                'row_count': response.get('results', {}).get('row_count', 0),
                'execution_time_ms': response.get('results', {}).get('execution_time_ms', 0),
                'total_time_ms': response.get('total_time_ms', 0),
                'error': response.get('error', None),
                'refusal_reason': response.get('refusal', {}).get('reason', None)
            }

            # Append to JSONL file
            with open(self._history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')

            logger.debug(f"Logged Q&A to history: {self._history_file}")

        except Exception as e:
            logger.warning(f"Failed to log Q&A history: {e}")

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent Q&A history

        Args:
            limit: Maximum entries to return

        Returns:
            List of history entries (newest first)
        """
        if not self._history_file.exists():
            return []

        entries = []
        try:
            with open(self._history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

            # Return newest first, limited
            return list(reversed(entries[-limit:]))

        except Exception as e:
            logger.warning(f"Failed to read Q&A history: {e}")
            return []

    def clear_history(self) -> bool:
        """Clear Q&A history file"""
        try:
            if self._history_file.exists():
                self._history_file.unlink()
                logger.info("Q&A history cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return False


# Convenience function
def ask(question: str, execute: bool = True) -> Dict[str, Any]:
    """
    Quick function to answer a question

    Args:
        question: Natural language question
        execute: Whether to execute SQL

    Returns:
        Answer response dict
    """
    parser = QuestionParser()
    return parser.answer(question, execute=execute)
