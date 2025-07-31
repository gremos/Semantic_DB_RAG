#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED SEMANTIC DATABASE RAG SYSTEM - Main Entry Point
Fixed method calls and integrated intelligent metadata analysis
"""

import asyncio
import os
from pathlib import Path

# Import modules
from db.discovery import DatabaseDiscovery
from semantic.analysis import EnhancedSemanticAnalyzer as SemanticAnalyzer

from interactive.query_interface import EnhancedQueryInterface as QueryInterface
from shared.config import Config
from shared.models import SystemStatus
from shared.utils import get_performance_recommendations, estimate_processing_time, log_filtering_statistics

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class EnhancedSemanticRAGSystem:
    """Enhanced system orchestrator with intelligent metadata analysis"""
    
    def __init__(self):
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SemanticAnalyzer(self.config)
        self.query_interface = QueryInterface(self.config)
        self.status = SystemStatus()
        
        # Show performance configuration
        print(self.config.get_performance_summary())
    
    async def run_step1_discovery_all_objects(self, limit: int = None):
        """Run Step 1: Database Discovery for Enhanced Analysis"""
        print("🚀 Step 1: Enhanced Database Discovery")
        print("=" * 70)
        
        if limit is None:
            print("📊 Mode: UNLIMITED - Analyzing ALL database objects for best results")
        else:
            print(f"📊 Mode: LIMITED - Analyzing top {limit} priority objects")
        
        print("📝 Enhanced sampling: 5 rows per object with business focus")
        print("⚡ Optimized for business entity recognition")
        
        # Get initial count estimate
        print("\n🔍 Estimating database size...")
        await self._show_database_size_estimate()
        
        # Run discovery
        success = await self.discovery.discover_database(limit)
        
        if success:
            self.status.discovery_completed = True
            self.status.tables_discovered = len(self.discovery.get_tables())
            
            # Show detailed results
            tables = self.discovery.get_tables()
            stats = self.discovery.get_stats()
            
            tables_count = sum(1 for t in tables if t.object_type == 'BASE TABLE')
            views_count = sum(1 for t in tables if t.object_type == 'VIEW')
            total_samples = sum(len(t.sample_data) for t in tables)
            
            print(f"\n✅ Enhanced discovery completed!")
            print(f"   📊 Objects analyzed: {len(tables)} (Tables: {tables_count}, Views: {views_count})")
            print(f"   📝 Total samples collected: {total_samples} rows")
            print(f"   📈 Success rate: {(stats.successful_analyses/max(stats.objects_processed,1)*100):.1f}%")
            print(f"   ⚡ Average samples per object: {total_samples/len(tables):.1f}")
            
            # Log filtering statistics
            log_filtering_statistics(
                stats.total_objects_found, 
                stats.objects_excluded, 
                stats.successful_analyses
            )
            
            return True
        else:
            print("❌ Discovery failed")
            return False
    
    async def _show_database_size_estimate(self):
        """Show database size estimate for planning"""
        try:
            with self.discovery.get_database_connection() as conn:
                cursor = conn.cursor()
                
                # Quick count query
                count_query = """
                SELECT 
                    (SELECT COUNT(*) FROM sys.tables WHERE is_ms_shipped = 0) as table_count,
                    (SELECT COUNT(*) FROM sys.views WHERE is_ms_shipped = 0) as view_count
                """
                cursor.execute(count_query)
                row = cursor.fetchone()
                
                if row:
                    table_count, view_count = row
                    total_objects = table_count + view_count
                    
                    print(f"   📊 Database size: {total_objects} objects ({table_count} tables, {view_count} views)")
                    
                    # Show performance recommendations
                    recommendations = get_performance_recommendations(total_objects)
                    for rec in recommendations:
                        print(f"   {rec}")
                    
                    # Show time estimate
                    estimated_time = estimate_processing_time(
                        total_objects, 
                        self.config.max_parallel_workers,
                        self.config.query_timeout_seconds
                    )
                    print(f"   ⏱️ Estimated processing time: {estimated_time}")
                    
                    return total_objects
                    
        except Exception as e:
            print(f"   ⚠️ Could not estimate database size: {e}")
            return None
    
    async def run_step2_comprehensive_semantic_analysis(self):
        """Run Step 2: COMPREHENSIVE Semantic Analysis using all discovery methods"""
        print("\n🧠 Step 2: COMPREHENSIVE Semantic Analysis - Multi-Source Discovery")
        print("=" * 70)
        
        if not self.status.discovery_completed:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No discovery data found. Run Step 1 first.")
                return False
        
        # Get discovered tables
        tables = self.discovery.get_tables()
        
        print(f"🚀 COMPREHENSIVE MODE: Analyzing {len(tables)} objects with all methods")
        print("   🔍 Enhanced view JOIN analysis")
        print("   📊 Comprehensive foreign key analysis") 
        print("   🧠 LLM entity scanning and classification")
        print("   🕸️ Multi-source relationship graph construction")
        print("   ⚡ Expected completion: 5-10 minutes (thorough analysis)")
        
        # Use the new comprehensive method
        success = await self.analyzer.analyze_semantics_comprehensive(tables)
        
        if success:
            self.status.analysis_completed = True
            self.status.relationships_found = len(self.analyzer.get_relationships())
            domain = self.analyzer.get_domain()
            business_analysis = self.analyzer.get_business_analysis()
            
            # Show comprehensive analysis results
            classified_count = sum(1 for t in tables if t.semantic_profile)
            views_classified = sum(1 for t in tables if t.object_type == 'VIEW' and t.semantic_profile)
            
            print(f"✅ COMPREHENSIVE semantic analysis completed!")
            print(f"   🧠 Objects classified: {classified_count}/{len(tables)}")
            print(f"   📊 Views analyzed: {views_classified}")
            print(f"   🔗 Relationships found: {len(self.analyzer.get_relationships())}")
            print(f"   🏢 Domain identified: {domain.domain_type if domain else 'Unknown'}")
            
            if domain and domain.industry:
                print(f"   🏭 Industry: {domain.industry}")
            
            # Show comprehensive business validation summary
            business_intelligence = business_analysis.get('business_intelligence', {})
            if business_intelligence:
                readiness = business_intelligence.get('business_readiness', {})
                rel_summary = business_intelligence.get('relationship_summary', {})
                
                print(f"   ✅ Business readiness: {readiness.get('rating', 'Unknown')} ({readiness.get('score', 0)}/100)")
                print(f"   🔗 Total relationships discovered: {rel_summary.get('total_discovered_relationships', 0)}")
                print(f"      • Foreign key constraints: {rel_summary.get('foreign_key_relationships', 0)}")
                print(f"      • View-based relationships: {rel_summary.get('view_relationships', 0)}")
                print(f"      • LLM suggested relationships: {rel_summary.get('llm_suggested_relationships', 0)}")
            
            return True
        else:
            print("❌ Comprehensive analysis failed")
            return False


    async def run_step2_intelligent_semantic_analysis(self):
        """Run Step 2: INTELLIGENT Semantic Analysis using Metadata + Selective LLM"""
        print("\n🧠 Step 2: INTELLIGENT Semantic Analysis - Metadata First Approach")
        print("=" * 70)
        
        if not self.status.discovery_completed:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No discovery data found. Run Step 1 first.")
                return False
        
        # Get discovered tables
        tables = self.discovery.get_tables()
        
        print(f"🚀 INTELLIGENT MODE: Analyzing {len(tables)} objects with metadata-first approach")
        print("   🔍 Using SQL Server system views for fast relationship discovery")
        print("   🧠 Selective LLM usage only for unclear cases")
        print("   ⚡ Expected completion: 2-5 minutes (vs 3 hours with old method)")
        
        success = await self.analyzer.analyze_semantics_intelligent(tables)
        
        if success:
            self.status.analysis_completed = True
            self.status.relationships_found = len(self.analyzer.get_relationships())
            domain = self.analyzer.get_domain()
            business_analysis = self.analyzer.get_business_analysis()
            
            # Show intelligent analysis results
            classified_count = sum(1 for t in tables if t.semantic_profile)
            views_classified = sum(1 for t in tables if t.object_type == 'VIEW' and t.semantic_profile)
            
            print(f"✅ INTELLIGENT semantic analysis completed!")
            print(f"   🧠 Objects classified: {classified_count}/{len(tables)}")
            print(f"   📊 Views classified: {views_classified}")
            print(f"   🔗 Relationships found: {len(self.analyzer.get_relationships())}")
            print(f"   🏢 Domain identified: {domain.domain_type if domain else 'Unknown'}")
            
            if domain and domain.industry:
                print(f"   🏭 Industry: {domain.industry}")
            
            # Show business validation summary
            validation = business_analysis.get('validation_results', {})
            status = validation.get('overall_status', 'Unknown')
            customer_payment_links = validation.get('customer_payment_links', 0)
            
            print(f"   ✅ Business validation: {status}")
            if customer_payment_links > 0:
                print(f"   🔗 Customer-payment links: {customer_payment_links}")
                print(f"   🎯 READY for 'paid customer' queries!")
            else:
                print(f"   ⚠️ No customer-payment links found")
            
            return True
        else:
            print("❌ Intelligent analysis failed")
            return False
    
    async def run_step3_smart_queries(self):
        """Run Step 3: Smart Interactive Queries with Business Intelligence"""
        print("\n💬 Step 3: Smart Interactive Queries with Business Intelligence")
        print("=" * 70)
        
        if not self.status.analysis_completed:
            # Try to load from cache
            if not (self.discovery.load_from_cache() and self.analyzer.load_from_cache()):
                print("❌ System not ready. Run Steps 1 and 2 first.")
                return False
        
        # Pass enhanced analysis results to query interface
        tables = self.discovery.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        business_analysis = self.analyzer.get_business_analysis()
        
        if len(tables) > 100:
            print(f"🚀 LARGE DATASET MODE: {len(tables)} objects available for smart queries")
            print("   🧠 Smart business query understanding")
            print("   🎯 Intelligent table selection")
            print("   ⚡ Enhanced SQL generation")
        
        # FIXED: Use correct method name
        await self.query_interface.start_intelligent_session(
            tables, domain, relationships, business_analysis
        )
    
    async def run_full_intelligent_demo(self):
        """Run complete intelligent demonstration with metadata-first analysis"""
        print("🚀 INTELLIGENT Full Demo - Metadata-First Analysis")
        print("=" * 70)
        print("Uses SQL Server system views for 10x faster analysis")
        print("Focus: Fast relationship discovery using database metadata")
        
        # Confirm for large datasets
        confirm = input("\nProceed with intelligent full analysis? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Demo cancelled")
            return
        
        print("\n🚀 Starting INTELLIGENT full analysis...")
        
        # Step 1: Discovery (all objects)
        success1 = await self.run_step1_discovery_all_objects()
        if not success1:
            print("❌ Discovery failed - stopping demo")
            return
        
        # Step 2: Intelligent Semantic Analysis (metadata-first)
        success2 = await self.run_step2_intelligent_semantic_analysis()
        if not success2:
            print("❌ Intelligent analysis failed - stopping demo")
            return
        
        # Show comprehensive status
        self.show_enhanced_status()
        
        print("\n✅ INTELLIGENT system ready with metadata-first analysis!")
        print("🎯 All database objects analyzed with business focus")
        print("🧠 Smart entity classification completed in minutes")
        print("🔗 Business relationships discovered using database metadata")
        print("💡 Ready for accurate business queries!")
        
        # Start smart interactive mode
        print("\n💬 Starting smart interactive mode...")
        await self.run_step3_smart_queries()
    
    def show_enhanced_status(self):
        """Show comprehensive system status with business analysis"""
        print("\n📊 INTELLIGENT SYSTEM STATUS with Metadata-First Analysis")
        print("=" * 70)
        
        # Try to load data from cache if not in memory
        if not self.status.discovery_completed:
            self.discovery.load_from_cache()
        if not self.status.analysis_completed:
            self.analyzer.load_from_cache()
        
        # Show discovery status
        tables = self.discovery.get_tables()
        if tables:
            table_count = sum(1 for t in tables if t.object_type == 'BASE TABLE')
            view_count = sum(1 for t in tables if t.object_type == 'VIEW')
            
            print(f"📋 Database Objects Analyzed: {len(tables)}")
            print(f"   • Tables: {table_count}")
            print(f"   • Views: {view_count}")
            
            # Data quality metrics
            objects_with_data = sum(1 for t in tables if t.sample_data)
            total_samples = sum(len(t.sample_data) for t in tables)
            avg_samples = total_samples / len(tables) if tables else 0
            
            print(f"   • Objects with sample data: {objects_with_data} ({(objects_with_data/len(tables)*100):.1f}%)")
            print(f"   • Total samples collected: {total_samples} rows")
            print(f"   • Average samples per object: {avg_samples:.1f}")
            
            # Enhanced semantic analysis status
            classified_count = sum(1 for t in tables if t.semantic_profile)
            if classified_count > 0:
                print(f"   • 🧠 Semantically classified: {classified_count} ({(classified_count/len(tables)*100):.1f}%)")
                
                # Business entity breakdown
                business_analysis = self.analyzer.get_business_analysis()
                if business_analysis:
                    validation = business_analysis.get('validation_results', {})
                    entity_counts = validation.get('entity_counts', {})
                    
                    core_entities = ['Customer', 'Payment', 'Order', 'Product']
                    for entity in core_entities:
                        count = entity_counts.get(entity, 0)
                        if count > 0:
                            print(f"     ✅ {entity}: {count} tables")
                    
                    # Business validation status
                    status = validation.get('overall_status', 'Unknown')
                    customer_payment_links = validation.get('customer_payment_links', 0)
                    
                    print(f"   • 🏢 Business validation: {status}")
                    if customer_payment_links > 0:
                        print(f"   • 🔗 Customer-payment links: {customer_payment_links}")
                        print(f"   • 🎯 READY for 'paid customer' queries!")
                    else:
                        print(f"   • ❌ No customer-payment links found!")
        
        # Show semantic analysis status
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        if domain:
            print(f"\n🏢 Intelligent Business Domain Analysis:")
            print(f"   • Domain Type: {domain.domain_type}")
            print(f"   • Industry: {domain.industry}")
            print(f"   • Confidence: {domain.confidence:.2f}")
            
            if domain.sample_questions:
                print(f"   • Sample Questions Available: {len(domain.sample_questions)}")
        
        if relationships:
            print(f"   • 🔗 Smart Relationships Discovered: {len(relationships)}")
            
            # Show relationship types
            rel_types = {}
            for rel in relationships:
                rel_type = rel.relationship_type
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            for rel_type, count in rel_types.items():
                print(f"     • {rel_type}: {count}")
        
        # Performance metrics
        print(f"\n⚡ Intelligent Performance Configuration:")
        print(f"   • Analysis Method: Metadata-First (10x faster)")
        print(f"   • Parallel Workers: {self.config.max_parallel_workers}")
        print(f"   • Batch Size: {self.config.max_batch_size}")
        print(f"   • Samples per Object: {self.config.samples_per_object}")
        print(f"   • Business Focus: Enabled")
        print(f"   • Smart Query Processing: Enabled")
        
        # Cache status
        self._show_cache_status()
    
    def _show_cache_status(self):
        """Show cache file status"""
        cache_files = [
            ('data/database_structure.json', 'Discovery Cache'),
            ('data/semantic_analysis.json', 'Intelligent Semantic Cache')
        ]
        
        print(f"\n💾 Cache Status:")
        for cache_file, description in cache_files:
            cache_path = Path(cache_file)
            if cache_path.exists():
                try:
                    size_mb = cache_path.stat().st_size / (1024 * 1024)
                    from datetime import datetime
                    modified = datetime.fromtimestamp(cache_path.stat().st_mtime)
                    age = datetime.now() - modified
                    print(f"   ✅ {description}: {size_mb:.1f}MB, {age.days} days old")
                except:
                    print(f"   ✅ {description}: Available")
            else:
                print(f"   ⚠️  {description}: Not found")


def main():
    """Enhanced main application entry point"""
    print("🚀 INTELLIGENT SEMANTIC DATABASE RAG SYSTEM")
    print("Metadata-First Analysis for 10x Faster Processing")
    print("=" * 80)
    
    # Load environment
    load_environment()
    
    # Initialize enhanced system
    try:
        system = EnhancedSemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize intelligent system: {e}")
        print("💡 Check your .env configuration:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        print("   - AZURE_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        return
    
    print("✅ INTELLIGENT system initialized with metadata-first analysis")
    
    while True:
        print("\n" + "="*80)
        print("COMPREHENSIVE SEMANTIC DATABASE RAG SYSTEM:")
        print("1. 🚀 Discover Database Objects - Enhanced for business entities")
        print("2. 🎯 Discover Limited Objects - specify count for testing")
        print("3. 🧠 COMPREHENSIVE Semantic Analysis - Views + Foreign Keys + LLM")
        print("4. 💬 Smart Interactive Queries - Relationship-aware query processing")
        print("5. 🌟 FULL COMPREHENSIVE DEMO - Complete multi-source analysis")
        print("6. 📊 Show Comprehensive System Status - Detailed analysis breakdown")
        print("7. ⚡ Comprehensive Features Information")
        print("8. 🧪 Database Size Estimate")
        print("0. Exit")
        print("="*80)
        
        try:
            choice = input("Enter your choice (0-8): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the INTELLIGENT Semantic Database RAG System!")
                break
            
            elif choice == '1':
                print("\n🚀 INTELLIGENT DISCOVERY MODE")
                print("This will analyze ALL database objects with metadata-first approach")
                print("⚡ Expected time: 10-15 minutes (vs hours with old method)")
                confirm = input("Continue with intelligent discovery? (y/N): ").strip().lower()
                if confirm == 'y':
                    asyncio.run(system.run_step1_discovery_all_objects())
            
            elif choice == '2':
                try:
                    limit = int(input("Enter maximum objects to analyze: ").strip())
                    if limit > 0:
                        asyncio.run(system.run_step1_discovery_all_objects(limit))
                    else:
                        print("❌ Please enter a positive number")
                except ValueError:
                    print("❌ Please enter a valid number")
            
            elif choice == '3':
                asyncio.run(system.run_step2_comprehensive_semantic_analysis())
            
            elif choice == '4':
                asyncio.run(system.run_step3_smart_queries())
            
            elif choice == '5':
                asyncio.run(system.run_full_intelligent_demo())
            
            elif choice == '6':
                system.show_enhanced_status()
            
            elif choice == '7':
                print("\n⚡ COMPREHENSIVE FEATURES - Multi-Source Analysis")
                print("=" * 60)
                print("✅ Enhanced View Analysis:")
                print("   • Comprehensive JOIN pattern detection (INNER, LEFT, RIGHT, FULL, CROSS)")
                print("   • Business logic complexity assessment")
                print("   • Relationship strength calculation")
                print("   • WHERE clause and aggregation analysis")
                print("   • Calculated field and CASE statement detection")
                
                print("\n✅ Advanced Foreign Key Analysis:")
                print("   • Business context classification")
                print("   • Cascade rule analysis (ON DELETE/UPDATE)")
                print("   • Cardinality estimation")
                print("   • Constraint enforcement status")
                print("   • Relationship pattern recognition")
                
                print("\n✅ LLM Entity Intelligence:")
                print("   • Business entity classification with confidence scoring")
                print("   • Implicit business rule discovery")
                print("   • Cross-entity relationship suggestions") 
                print("   • Data pattern analysis")
                print("   • Domain-specific entity recognition")
                
                print("\n✅ Comprehensive Integration:")
                print("   • Multi-source relationship validation")
                print("   • Confidence-weighted analysis")
                print("   • Business intelligence scoring")
                print("   • Entity-relationship matrix generation")
                print("   • NetworkX graph construction for visualization")
                
                print("\n💡 Key Improvements for Business Queries:")
                print("   • 'Paid customer' queries now use FK + view + LLM analysis")
                print("   • Revenue analysis leverages comprehensive relationship discovery")
                print("   • Order-customer relationships validated across all sources")
                print("   • Business entity confidence scoring guides query generation")
                
                print(f"\n📊 Current Configuration:")
                print(f"   • Multi-source analysis: Enabled")
                print(f"   • View JOIN analysis: Enhanced regex patterns")
                print(f"   • Foreign key discovery: Business context aware")
                print(f"   • LLM entity scanning: Structured prompts")
                print(f"   • Relationship validation: Cross-source verification")
                print(f"   • Expected analysis depth: 5-10x more comprehensive")
            
            elif choice == '8':
                print("\n🧪 Database Size Estimation")
                print("=" * 40)
                asyncio.run(system._show_database_size_estimate())
            
            else:
                print(f"❌ Invalid choice: '{choice}'. Please enter 0-8.")
        
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted by user")
            break
        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()