#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIGH-PERFORMANCE SEMANTIC DATABASE RAG SYSTEM - Main Entry Point
Optimized for analyzing 500+ database objects with 5 samples each
"""

import asyncio
import os
from pathlib import Path

# Import modules
from db.discovery import DatabaseDiscovery
from semantic.analysis import SemanticAnalyzer
from interactive.query_interface import QueryInterface
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

class HighPerformanceSemanticRAGSystem:
    """HIGH-PERFORMANCE system orchestrator for 500+ objects"""
    
    def __init__(self):
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SemanticAnalyzer(self.config)
        self.query_interface = QueryInterface(self.config)
        self.status = SystemStatus()
        
        # Show performance configuration
        print(self.config.get_performance_summary())
    
    async def run_step1_discovery_all_objects(self, limit: int = None):
        """Run Step 1: HIGH-PERFORMANCE Database Discovery for ALL Objects"""
        print("🚀 Step 1: HIGH-PERFORMANCE Database Discovery (ALL Objects)")
        print("=" * 70)
        
        if limit is None:
            print("📊 Mode: UNLIMITED - Analyzing ALL database objects (500+)")
        else:
            print(f"📊 Mode: LIMITED - Analyzing top {limit} objects")
        
        print("📝 Samples per object: 5 rows (as requested)")
        print("⚡ Using aggressive parallelism and FAST queries")
        
        # Get initial count estimate for performance planning
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
            
            print(f"\n✅ HIGH-PERFORMANCE discovery completed!")
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
    
    async def run_step2_analysis_enhanced(self):
        """Run Step 2: Enhanced Semantic Analysis for Large Datasets"""
        print("\n🧠 Step 2: Enhanced Semantic Analysis (Large Dataset Support)")
        print("=" * 70)
        
        if not self.status.discovery_completed:
            # Try to load from cache
            if not self.discovery.load_from_cache():
                print("❌ No discovery data found. Run Step 1 first.")
                return False
        
        # Pass discovery results to analyzer
        tables = self.discovery.get_tables()
        
        if len(tables) > 100:
            print(f"🚀 LARGE DATASET MODE: Analyzing {len(tables)} objects")
            print("   ⚡ Using optimized batch processing")
            print("   🧠 Enhanced AI classification for views")
        
        success = await self.analyzer.analyze_semantics(tables)
        
        if success:
            self.status.analysis_completed = True
            self.status.relationships_found = len(self.analyzer.get_relationships())
            domain = self.analyzer.get_domain()
            
            # Show detailed analysis results
            classified_count = sum(1 for t in tables if t.semantic_profile)
            views_classified = sum(1 for t in tables if t.object_type == 'VIEW' and t.semantic_profile)
            
            print(f"✅ Enhanced semantic analysis completed!")
            print(f"   🧠 Objects classified: {classified_count}/{len(tables)}")
            print(f"   📊 Views classified: {views_classified} (major improvement)")
            print(f"   🔗 Relationships found: {len(self.analyzer.get_relationships())}")
            print(f"   🏢 Domain identified: {domain.domain_type if domain else 'Unknown'}")
            
            if domain and domain.industry:
                print(f"   🏭 Industry: {domain.industry}")
            
            return True
        else:
            print("❌ Analysis failed")
            return False
    
    async def run_step3_queries_enhanced(self):
        """Run Step 3: Enhanced Interactive Queries with Large Dataset Support"""
        print("\n💬 Step 3: Enhanced Interactive Queries (Large Dataset Support)")
        print("=" * 70)
        
        if not self.status.analysis_completed:
            # Try to load from cache
            if not (self.discovery.load_from_cache() and self.analyzer.load_from_cache()):
                print("❌ System not ready. Run Steps 1 and 2 first.")
                return False
        
        # Pass semantic analysis results to query interface
        tables = self.discovery.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        if len(tables) > 100:
            print(f"🚀 LARGE DATASET MODE: {len(tables)} objects available for queries")
            print("   💡 Enhanced table relevance scoring")
            print("   🧠 Improved SQL generation with view support")
        
        await self.query_interface.start_interactive_session(tables, domain, relationships)
    
    async def run_full_demo_high_performance(self):
        """Run complete high-performance demonstration"""
        print("🚀 HIGH-PERFORMANCE Full Demo (500+ Objects)")
        print("=" * 70)
        print("This will analyze ALL database objects with 5 samples each")
        print("Estimated time: 15-30 minutes for large databases")
        
        # Confirm for large datasets
        confirm = input("\nProceed with full analysis? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Demo cancelled")
            return
        
        print("\n🚀 Starting HIGH-PERFORMANCE full analysis...")
        
        # Step 1: Discovery (all objects)
        success1 = await self.run_step1_discovery_all_objects()
        if not success1:
            print("❌ Discovery failed - stopping demo")
            return
        
        # Step 2: Semantic Analysis
        success2 = await self.run_step2_analysis_enhanced()
        if not success2:
            print("❌ Analysis failed - stopping demo")
            return
        
        # Show comprehensive status
        self.show_enhanced_status()
        
        print("\n✅ HIGH-PERFORMANCE system ready!")
        print("🎯 All database objects analyzed with 5 samples each")
        print("🧠 Semantic classification completed")
        print("🔗 Relationships discovered")
        
        # Start interactive mode
        print("\n💬 Starting enhanced interactive mode...")
        await self.run_step3_queries_enhanced()
    
    def show_enhanced_status(self):
        """Show comprehensive system status for large datasets"""
        print("\n📊 HIGH-PERFORMANCE SYSTEM STATUS")
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
            
            # Semantic analysis status
            classified_count = sum(1 for t in tables if t.semantic_profile)
            if classified_count > 0:
                print(f"   • 🧠 Semantically classified: {classified_count} ({(classified_count/len(tables)*100):.1f}%)")
                
                # Views classification (major improvement)
                views_classified = sum(1 for t in tables if t.object_type == 'VIEW' and t.semantic_profile)
                if view_count > 0:
                    print(f"   • 📊 Views classified: {views_classified}/{view_count} (improved from 0)")
        
        # Show semantic analysis status
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        if domain:
            print(f"\n🏢 Business Domain Analysis:")
            print(f"   • Domain Type: {domain.domain_type}")
            print(f"   • Industry: {domain.industry}")
            print(f"   • Confidence: {domain.confidence:.2f}")
            print(f"   • Key Entities: {', '.join(domain.entities[:5])}")
        
        if relationships:
            print(f"   • 🔗 Relationships Discovered: {len(relationships)}")
        
        # Performance metrics
        print(f"\n⚡ Performance Configuration:")
        print(f"   • Parallel Workers: {self.config.max_parallel_workers}")
        print(f"   • Batch Size: {self.config.max_batch_size}")
        print(f"   • Samples per Object: {self.config.samples_per_object}")
        print(f"   • FAST Queries: {'Enabled' if self.config.use_fast_queries else 'Disabled'}")
        
        # Cache status
        self._show_cache_status()
    
    def _show_cache_status(self):
        """Show cache file status"""
        cache_files = [
            ('data/database_structure.json', 'Discovery Cache'),
            ('data/semantic_analysis.json', 'Semantic Cache')
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
    """HIGH-PERFORMANCE main application entry point"""
    print("🚀 HIGH-PERFORMANCE SEMANTIC DATABASE RAG SYSTEM")
    print("Optimized for 500+ Objects with 5 Samples Each")
    print("=" * 80)
    
    # Load environment
    load_environment()
    
    # Initialize system
    try:
        system = HighPerformanceSemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("💡 Check your .env configuration:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        print("   - AZURE_ENDPOINT")
        print("   - DEPLOYMENT_NAME")
        return
    
    print("✅ HIGH-PERFORMANCE system initialized successfully")
    
    while True:
        print("\n" + "="*80)
        print("HIGH-PERFORMANCE MENU OPTIONS:")
        print("1. 🚀 Discover ALL Objects (500+) - 5 samples each")
        print("2. 🎯 Discover Limited Objects - specify count")
        print("3. 🧠 Semantic Analysis - enhanced for large datasets")
        print("4. 💬 Interactive Queries - improved table selection")
        print("5. 🌟 FULL DEMO - complete analysis of all objects")
        print("6. 📊 Show Enhanced System Status")
        print("7. ⚡ Performance Information & Recommendations")
        print("8. 🧪 Database Size Estimate")
        print("0. Exit")
        print("="*80)
        
        try:
            choice = input("Enter your choice (0-8): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the HIGH-PERFORMANCE Semantic Database RAG System!")
                break
            
            elif choice == '1':
                print("\n🚀 UNLIMITED DISCOVERY MODE")
                print("This will analyze ALL database objects (500+) with 5 samples each")
                print("⚠️  This may take 15-30 minutes for large databases")
                confirm = input("Continue? (y/N): ").strip().lower()
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
                asyncio.run(system.run_step2_analysis_enhanced())
            
            elif choice == '4':
                asyncio.run(system.run_step3_queries_enhanced())
            
            elif choice == '5':
                asyncio.run(system.run_full_demo_high_performance())
            
            elif choice == '6':
                system.show_enhanced_status()
            
            elif choice == '7':
                print("\n⚡ HIGH-PERFORMANCE FEATURES & RECOMMENDATIONS")
                print("=" * 60)
                print("✅ Optimizations Applied:")
                print("   • Removed artificial object limits (analyze ALL 500+ objects)")
                print("   • Increased parallel workers for faster processing")
                print("   • FAST query option for 2-5x speed improvement")
                print("   • Enhanced view analysis (previous versions showed 0 views)")
                print("   • 5 samples per object (as requested)")
                print("   • Minimal filtering - keeps business objects")
                print("   • Large dataset mode with optimized batching")
                print("   • Connection pooling and query optimization")
                
                print("\n💡 Performance Recommendations:")
                print("   • Run discovery during off-peak hours for large databases")
                print("   • Use cached results for repeated analysis")
                print("   • Monitor system resources during large discoveries")
                print("   • Consider increasing parallel workers if system can handle it")
                
                print(f"\n📊 Current Configuration:")
                config = system.config
                print(f"   • Max parallel workers: {config.max_parallel_workers}")
                print(f"   • Batch size: {config.max_batch_size}")
                print(f"   • Samples per object: {config.samples_per_object}")
                print(f"   • Query timeout: {config.query_timeout_seconds}s")
                print(f"   • Rate limit delay: {config.rate_limit_delay}s")
            
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