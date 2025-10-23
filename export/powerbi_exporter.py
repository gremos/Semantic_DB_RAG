"""
Export semantic model as Power BI Tabular Model (Analysis Services compatible).
This creates a .bim file that can be imported into Power BI Desktop or Analysis Services.
"""

from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)

class PowerBIExporter:
    """Export semantic model to Power BI Tabular Model format."""
    
    @staticmethod
    def export(semantic_model: Dict[str, Any], output_file: str = "semantic_model.bim") -> bool:
        """
        Export semantic model as Power BI .bim file.
        
        Args:
            semantic_model: Semantic model JSON
            output_file: Output filename (.bim extension)
        
        Returns:
            Success boolean
        """
        try:
            logger.info(f"Exporting to Power BI Tabular Model: {output_file}")
            
            # Build Tabular Model structure
            tabular_model = {
                "name": "SemanticModel",
                "compatibilityLevel": 1600,  # Power BI / SQL Server 2022
                "model": {
                    "culture": "en-US",
                    "dataSources": [PowerBIExporter._create_data_source(semantic_model)],
                    "tables": PowerBIExporter._create_tables(semantic_model),
                    "relationships": PowerBIExporter._create_relationships(semantic_model)
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(tabular_model, f, indent=2)
            
            logger.info(f"✅ Power BI model exported: {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export Power BI model: {e}", exc_info=True)
            return False
    
    @staticmethod
    def _create_data_source(semantic_model: Dict[str, Any]) -> Dict[str, Any]:
        """Create data source definition."""
        dialect = semantic_model.get("audit", {}).get("dialect", "tsql")
        
        # Get connection info from first source asset
        source_assets = semantic_model.get("audit", {}).get("source_assets_used", [])
        first_table = source_assets[0]["name_or_path"] if source_assets else "dbo.Unknown"
        schema = first_table.split('.')[0] if '.' in first_table else "dbo"
        
        return {
            "type": "structured",
            "name": "DatabaseConnection",
            "connectionDetails": {
                "protocol": "tds",
                "address": {
                    "server": "localhost",  # User needs to update this
                    "database": "YourDatabase"  # User needs to update this
                },
                "authentication": "windows"
            }
        }
    
    @staticmethod
    def _create_tables(semantic_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create table definitions."""
        tables = []
        
        # Add fact tables
        for fact in semantic_model.get("facts", []):
            tables.append(PowerBIExporter._create_fact_table(fact))
        
        # Add dimension tables
        for dimension in semantic_model.get("dimensions", []):
            tables.append(PowerBIExporter._create_dimension_table(dimension))
        
        # Add entity tables
        for entity in semantic_model.get("entities", []):
            tables.append(PowerBIExporter._create_entity_table(entity))
        
        return tables
    
    @staticmethod
    def _create_fact_table(fact: Dict[str, Any]) -> Dict[str, Any]:
        """Create fact table definition with measures."""
        table = {
            "name": fact["name"],
            "description": fact.get("description", ""),
            "columns": PowerBIExporter._create_columns(fact.get("columns", [])),
            "measures": PowerBIExporter._create_measures(fact.get("measures", [])),
            "partitions": [{
                "name": "Partition",
                "source": {
                    "type": "m",
                    "expression": [
                        f"let",
                        f"    Source = Sql.Database(\"localhost\", \"YourDatabase\"),",
                        f"    Table = Source{{[Schema=\"{fact['source'].split('.')[0]}\",Item=\"{fact['source'].split('.')[1]}\"]}}"[1],
                        f"in",
                        f"    Table"
                    ]
                }
            }]
        }
        
        return table
    
    @staticmethod
    def _create_dimension_table(dimension: Dict[str, Any]) -> Dict[str, Any]:
        """Create dimension table definition."""
        schema, table_name = dimension["source"].split('.') if '.' in dimension["source"] else ("dbo", dimension["source"])
        
        return {
            "name": dimension["name"],
            "description": f"Dimension: {dimension['name']}",
            "columns": PowerBIExporter._create_columns(dimension.get("columns", [])),
            "partitions": [{
                "name": "Partition",
                "source": {
                    "type": "m",
                    "expression": [
                        f"let",
                        f"    Source = Sql.Database(\"localhost\", \"YourDatabase\"),",
                        f"    Table = Source{{[Schema=\"{schema}\",Item=\"{table_name}\"]}}"[1],
                        f"in",
                        f"    Table"
                    ]
                }
            }]
        }
    
    @staticmethod
    def _create_entity_table(entity: Dict[str, Any]) -> Dict[str, Any]:
        """Create entity table definition."""
        schema, table_name = entity["source"].split('.') if '.' in entity["source"] else ("dbo", entity["source"])
        
        return {
            "name": entity["name"],
            "description": entity.get("description", ""),
            "columns": PowerBIExporter._create_columns(entity.get("columns", [])),
            "partitions": [{
                "name": "Partition",
                "source": {
                    "type": "m",
                    "expression": [
                        f"let",
                        f"    Source = Sql.Database(\"localhost\", \"YourDatabase\"),",
                        f"    Table = Source{{[Schema=\"{schema}\",Item=\"{table_name}\"]}}"[1],
                        f"in",
                        f"    Table"
                    ]
                }
            }]
        }
    
    @staticmethod
    def _create_columns(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create column definitions."""
        pbi_columns = []
        
        for col in columns:
            pbi_columns.append({
                "name": col["name"],
                "dataType": PowerBIExporter._map_datatype(col.get("type", "string")),
                "sourceColumn": col["name"],
                "summarizeBy": "none",
                "annotations": [{
                    "name": "SemanticRole",
                    "value": col.get("semantic_role", "")
                }, {
                    "name": "Description",
                    "value": col.get("description", "")
                }]
            })
        
        return pbi_columns
    
    @staticmethod
    def _create_measures(measures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create measure definitions."""
        pbi_measures = []
        
        for measure in measures:
            # Convert SQL expression to DAX
            dax_expression = PowerBIExporter._sql_to_dax(measure.get("expression", ""))
            
            pbi_measures.append({
                "name": measure["name"],
                "expression": dax_expression,
                "formatString": PowerBIExporter._get_format_string(measure.get("format", "number")),
                "description": measure.get("description", ""),
                "annotations": [{
                    "name": "DependsOn",
                    "value": ", ".join(measure.get("depends_on", []))
                }, {
                    "name": "FiltersApplied",
                    "value": ", ".join(measure.get("filters_applied", []))
                }]
            })
        
        return pbi_measures
    
    @staticmethod
    def _create_relationships(semantic_model: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create relationship definitions."""
        pbi_relationships = []
        
        for rel in semantic_model.get("relationships", []):
            pbi_relationships.append({
                "name": f"{rel['from']}_to_{rel['to']}",
                "fromTable": rel["from"],
                "fromColumn": "ID",  # Simplified - should extract from FK
                "toTable": rel["to"],
                "toColumn": "ID",
                "crossFilteringBehavior": "oneDirection",
                "cardinality": PowerBIExporter._map_cardinality(rel.get("cardinality", "many-to-one"))
            })
        
        return pbi_relationships
    
    @staticmethod
    def _map_datatype(sql_type: str) -> str:
        """Map SQL datatype to Power BI datatype."""
        type_lower = sql_type.lower()
        
        if 'int' in type_lower:
            return "int64"
        elif 'decimal' in type_lower or 'numeric' in type_lower or 'money' in type_lower:
            return "decimal"
        elif 'float' in type_lower or 'real' in type_lower:
            return "double"
        elif 'date' in type_lower or 'time' in type_lower:
            return "dateTime"
        elif 'bit' in type_lower or 'bool' in type_lower:
            return "boolean"
        else:
            return "string"
    
    @staticmethod
    def _map_cardinality(cardinality: str) -> str:
        """Map semantic cardinality to Power BI cardinality."""
        if "many-to-one" in cardinality.lower():
            return "manyToOne"
        elif "one-to-many" in cardinality.lower():
            return "oneToMany"
        else:
            return "oneToOne"
    
    @staticmethod
    def _get_format_string(format_type: str) -> str:
        """Get Power BI format string."""
        formats = {
            "currency": "$#,##0.00",
            "number": "#,##0",
            "percentage": "0.00%",
            "duration": "hh:mm:ss"
        }
        return formats.get(format_type, "#,##0")
    
    @staticmethod
    def _sql_to_dax(sql_expression: str) -> str:
        """
        Convert SQL expression to DAX (basic conversion).
        User will need to refine these.
        """
        # Basic conversions
        dax = sql_expression.replace("SUM(", "SUMX(")
        dax = dax.replace("COUNT(", "COUNTROWS(")
        dax = dax.replace("AVG(", "AVERAGEX(")
        
        # Add table context (simplified)
        # Real conversion would need proper parsing
        return f"// TODO: Verify DAX expression\n{dax}"