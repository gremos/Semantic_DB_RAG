#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENHANCED SEMANTIC DATABASE RAG SYSTEM - Main Entry Point
Integrated Smart Business Analysis for Accurate Query Processing
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
    """Enhanced system orchestrator with smart business analysis"""
    
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
    
    async def run_step2_enhanced_semantic_analysis(self):
        """Run Step 2: Enhanced Semantic Analysis with Smart Business Understanding"""
        print("\n🧠 Step 2: Enhanced Semantic Analysis with Smart Business Understanding")
        print("=" * 70)
        
        if not self.status.discovery_completed:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No discovery data found. Run Step 1 first.")
                return False
        
        # Pass discovery results to enhanced analyzer
        tables = self.discovery.get_tables()
        
        if len(tables) > 100:
            print(f"🚀 LARGE DATASET MODE: Enhanced analysis of {len(tables)} objects")
            print("   🧠 Smart business entity recognition")
            print("   🔗 Intelligent relationship discovery")
            print("   ✅ Business logic validation")
        
        success = await self.analyzer.analyze_semantics(tables)
        
        if success:
            self.status.analysis_completed = True
            self.status.relationships_found = len(self.analyzer.get_relationships())
            domain = self.analyzer.get_domain()
            business_analysis = self.analyzer.get_business_analysis()
            
            # Show enhanced analysis results
            classified_count = sum(1 for t in tables if t.semantic_profile)
            views_classified = sum(1 for t in tables if t.object_type == 'VIEW' and t.semantic_profile)
            
            print(f"✅ Enhanced semantic analysis completed!")
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
            else:
                print(f"   ⚠️ No customer-payment links found - paid customer queries may fail")
            
            return True
        else:
            print("❌ Enhanced analysis failed")
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
        
        await self.query_interface.start_interactive_session(
            tables, domain, relationships, business_analysis
        )
    
    async def run_full_enhanced_demo(self):
        """Run complete enhanced demonstration with smart business analysis"""
        print("🚀 ENHANCED Full Demo with Smart Business Analysis")
        print("=" * 70)
        print("This will analyze ALL database objects with enhanced business understanding")
        print("Focus: Customer-Payment relationships for accurate business queries")
        
        # Confirm for large datasets
        confirm = input("\nProceed with enhanced full analysis? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Demo cancelled")
            return
        
        print("\n🚀 Starting ENHANCED full analysis...")
        
        # Step 1: Discovery (all objects)
        success1 = await self.run_step1_discovery_all_objects()
        if not success1:
            print("❌ Discovery failed - stopping demo")
            return
        
        # Step 2: Enhanced Semantic Analysis
        success2 = await self.run_step2_enhanced_semantic_analysis()
        if not success2:
            print("❌ Enhanced analysis failed - stopping demo")
            return
        
        # Show comprehensive status
        self.show_enhanced_status()
        
        print("\n✅ ENHANCED system ready with smart business understanding!")
        print("🎯 All database objects analyzed with business focus")
        print("🧠 Smart entity classification completed")
        print("🔗 Business relationships discovered")
        print("💡 Ready for accurate business queries!")
        
        # Start smart interactive mode
        print("\n💬 Starting smart interactive mode...")
        await self.run_step3_smart_queries()
    
    def show_enhanced_status(self):
        """Show comprehensive system status with business analysis"""
        print("\n📊 ENHANCED SYSTEM STATUS with Smart Business Analysis")
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
                    else:
                        print(f"   • ❌ No customer-payment links found!")
        
        # Show semantic analysis status
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        if domain:
            print(f"\n🏢 Enhanced Business Domain Analysis:")
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
        print(f"\n⚡ Enhanced Performance Configuration:")
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
            ('data/semantic_analysis.json', 'Enhanced Semantic Cache')
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
    print("🚀 ENHANCED SEMANTIC DATABASE RAG SYSTEM")
    print("Smart Business Analysis for Accurate Query Processing")
    print("=" * 80)
    
    # Load environment
    load_environment()
    
    # Initialize enhanced system
    try:
        system = EnhancedSemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize enhanced system: {e}")
        print("💡 Check your .env configuration:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        print("   - AZURE_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        return
    
    print("✅ ENHANCED system initialized with smart business analysis")
    
    while True:
        print("\n" + "="*80)
        print("ENHANCED MENU OPTIONS - Smart Business Analysis:")
        print("1. 🚀 Discover Database Objects - Enhanced for business entities")
        print("2. 🎯 Discover Limited Objects - specify count for testing")
        print("3. 🧠 Enhanced Semantic Analysis - Smart business entity recognition")
        print("4. 💬 Smart Interactive Queries - Intelligent business query processing")
        print("5. 🌟 FULL ENHANCED DEMO - Complete smart business analysis")
        print("6. 📊 Show Enhanced System Status - Business entity breakdown")
        print("7. ⚡ Smart Features Information")
        print("8. 🧪 Database Size Estimate")
        print("0. Exit")
        print("="*80)
        
        try:
            choice = input("Enter your choice (0-8): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the ENHANCED Semantic Database RAG System!")
                break
            
            elif choice == '1':
                print("\n🚀 ENHANCED DISCOVERY MODE")
                print("This will analyze ALL database objects with business focus")
                print("⚠️  This may take 15-30 minutes for large databases")
                confirm = input("Continue with enhanced discovery? (y/N): ").strip().lower()
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
                asyncio.run(system.run_step2_enhanced_semantic_analysis())
            
            elif choice == '4':
                asyncio.run(system.run_step3_smart_queries())
            
            elif choice == '5':
                asyncio.run(system.run_full_enhanced_demo())
            
            elif choice == '6':
                system.show_enhanced_status()
            
            elif choice == '7':
                print("\n⚡ ENHANCED SMART FEATURES")
                print("=" * 60)
                print("✅ Smart Business Entity Recognition:")
                print("   • Advanced pattern matching for customers, payments, orders")
                print("   • LLM-powered business purpose identification")
                print("   • Confidence scoring and validation")
                
                print("\n✅ Intelligent Relationship Discovery:")
                print("   • Automatic customer-payment link detection")
                print("   • Business logic validation")
                print("   • Relationship confidence scoring")
                
                print("\n✅ Smart Query Processing:")
                print("   • Business intent analysis")
                print("   • Intelligent table selection")
                print("   • Context-aware SQL generation")
                print("   • Enhanced error handling and suggestions")
                
                print("\n💡 Key Improvements for 'Paid Customer' Queries:")
                print("   • Identifies customer and payment tables accurately")
                print("   • Discovers relationships between customer and payment data")
                print("   • Generates proper JOIN queries for business questions")
                print("   • Validates results and provides business interpretation")
                
                print("\n🎯 Business Query Examples That Now Work Better:")
                print("   • 'How many customers have paid in 2025?' - Smart customer-payment joins")
                print("   • 'What is our total revenue?' - Intelligent payment amount aggregation")
                print("   • 'Show paid customers' - Proper relationship understanding")
                
                print(f"\n📊 Current Configuration:")
                config = system.config
                print(f"   • Enhanced semantic analysis: Enabled")
                print(f"   • Smart query processing: Enabled")
                print(f"   • Business entity focus: Enabled")
                print(f"   • Relationship validation: Enabled")
            
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