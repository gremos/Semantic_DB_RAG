from typing import Dict, Any, Tuple, List
from llm.azure_client import AzureLLMClient
from validation.schema_validator import SchemaValidator
from discovery.discovery_compressor import DiscoveryCompressor
from .phases.table_classifier import TableClassifier
from .phases.measure_identifier import MeasureIdentifier
from .phases.status_analyzer import StatusColumnAnalyzer
from .phases.relationship_inferrer import RelationshipInferrer
from .phases.model_assembler import ModelAssembler
import logging
import time

logger = logging.getLogger(__name__)

class IncrementalModeler:
    """Incremental semantic model generation - one table/column at a time."""
    
    def __init__(self, llm_client: AzureLLMClient, validator: SchemaValidator):
        self.llm = llm_client
        self.validator = validator
        self.compressor = DiscoveryCompressor()
        
        # Initialize phase processors
        self.classifier = TableClassifier(llm_client)
        self.measure_identifier = MeasureIdentifier(llm_client)
        self.status_analyzer = StatusColumnAnalyzer(llm_client)
        self.relationship_inferrer = RelationshipInferrer(llm_client)
        self.assembler = ModelAssembler(llm_client)
    
    def create_model(
        self,
        compressed_discovery: Dict[str, Any],
        domain_hints: str = ""
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Create semantic model using incremental approach.
        
        Returns:
            (success, model_json, error)
        """
        try:
            logger.info("=== PHASE 1: TABLE CLASSIFICATION ===")
            classifications = self._phase1_classify_tables(compressed_discovery)
            logger.info(f"  Classified {len(classifications)} tables")
            
            logger.info("=== PHASE 2: MEASURE IDENTIFICATION ===")
            measures = self._phase2_identify_measures(classifications, compressed_discovery)
            logger.info(f"  Identified measures for {len(measures)} fact tables")
            
            logger.info("=== PHASE 3: STATUS COLUMN ANALYSIS ===")
            status_columns = self._phase3_analyze_status_columns(compressed_discovery)
            logger.info(f"  Analyzed {len(status_columns)} status columns")
            
            logger.info("=== PHASE 4: RELATIONSHIP INFERENCE ===")
            relationships = self._phase4_infer_relationships(compressed_discovery)
            logger.info(f"  Inferred {len(relationships)} relationships")
            
            logger.info("=== PHASE 5: MODEL ASSEMBLY ===")
            success, model, error = self.assembler.assemble_model(
                classifications,
                measures,
                status_columns,
                relationships,
                compressed_discovery
            )
            
            if not success:
                return (False, {}, error)
            
            # Validate
            logger.info("Validating semantic model...")
            valid, schema_error = self.validator.validate(model, "semantic_model")
            
            if not valid:
                logger.warning(f"Schema validation failed: {schema_error}")
                # Try to continue anyway
            
            logger.info("Semantic model created successfully via incremental approach")
            return (True, model, "")
        
        except Exception as e:
            logger.error(f"Incremental modeling failed: {e}", exc_info=True)
            return (False, {}, str(e))
    
    def _phase1_classify_tables(self, compressed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Phase 1: Classify all tables."""
        classifications = {}
        tables = list(compressed["tables"].keys())
        
        logger.info(f"  Classifying {len(tables)} tables...")
        
        for idx, table_name in enumerate(tables, 1):
            if idx % 20 == 0:
                logger.info(f"    Progress: {idx}/{len(tables)} tables classified")
            
            table_info = self.compressor.get_table_for_classification(compressed, table_name)
            if not table_info:
                continue
            
            success, result, error = self.classifier.classify_table(table_info)
            if success:
                classifications[table_name] = result
            
            # Small delay to avoid rate limits (optional)
            if idx % 10 == 0:
                time.sleep(0.5)
        
        return classifications
    
    def _phase2_identify_measures(
        self,
        classifications: Dict[str, Dict[str, Any]],
        compressed: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Phase 2: Identify measures for fact tables."""
        measures = {}
        
        # Get fact tables
        facts = [
            table_name for table_name, cls in classifications.items()
            if cls["classification"] == "fact"
        ]
        
        logger.info(f"  Identifying measures for {len(facts)} fact tables...")
        
        for idx, table_name in enumerate(facts, 1):
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(facts)} facts processed")
            
            columns = self.compressor.get_columns_for_table(compressed, table_name)
            
            success, table_measures, error = self.measure_identifier.identify_measures(
                table_name,
                columns
            )
            
            if success and table_measures:
                measures[table_name] = table_measures
            
            if idx % 5 == 0:
                time.sleep(0.5)
        
        return measures
    
    def _phase3_analyze_status_columns(
        self,
        compressed: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Phase 3: Analyze status indicator columns."""
        status_columns = {}
        column_samples = compressed.get("column_samples", {})
        
        # Scan all tables for status columns
        all_status = []
        for table_name, table_data in compressed["tables"].items():
            columns = table_data.get("columns", [])
            status_col_names = self.status_analyzer.identify_status_columns(columns)
            
            for col_name in status_col_names:
                all_status.append((table_name, col_name, columns))
        
        logger.info(f"  Analyzing {len(all_status)} status columns...")
        
        for idx, (table_name, col_name, columns) in enumerate(all_status, 1):
            if idx % 10 == 0:
                logger.info(f"    Progress: {idx}/{len(all_status)} status columns analyzed")
            
            # Find column info
            col_info = next((c for c in columns if c["name"] == col_name), None)
            if not col_info:
                continue
            
            # Get sample values
            sample_key = f"{table_name}.{col_name}"
            sample_values = column_samples.get(sample_key)
            
            success, result, error = self.status_analyzer.analyze_status_column(
                table_name,
                col_name,
                col_info["type"],
                col_info.get("nullable", True),
                sample_values
            )
            
            if success:
                status_columns[sample_key] = result
            
            if idx % 5 == 0:
                time.sleep(0.3)
        
        return status_columns
    
    def _phase4_infer_relationships(
        self,
        compressed: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Phase 4: Infer relationships from foreign keys."""
        relationships = []
        
        # Collect all FKs
        all_fks = []
        for table_name, table_data in compressed["tables"].items():
            fks = table_data.get("fks", [])
            for fk in fks:
                # Parse FK: "column→ref_table.ref_column"
                parts = fk.split('→')
                if len(parts) == 2:
                    fk_col = parts[0]
                    ref_parts = parts[1].split('.')
                    if len(ref_parts) == 2:
                        ref_table = ref_parts[0]
                        ref_col = ref_parts[1]
                        all_fks.append((table_name, fk_col, ref_table, ref_col))
        
        logger.info(f"  Inferring {len(all_fks)} relationships...")
        
        for idx, (from_table, fk_col, to_table, to_col) in enumerate(all_fks, 1):
            if idx % 20 == 0:
                logger.info(f"    Progress: {idx}/{len(all_fks)} relationships inferred")
            
            success, result, error = self.relationship_inferrer.infer_relationship(
                from_table,
                fk_col,
                to_table,
                to_col
            )
            
            if success:
                relationships.append(result)
            
            if idx % 10 == 0:
                time.sleep(0.5)
        
        return relationships