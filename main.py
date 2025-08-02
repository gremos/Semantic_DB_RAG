#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Semantic Database RAG System - Main Entry Point
Clean, readable, and maintainable implementation
"""

import asyncio
import os
from pathlib import Path

from db.discovery import DatabaseDiscovery
from semantic.analysis import SemanticAnalyzer
from interactive.query_interface import QueryInterface
from shared.config import Config

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')

class SemanticRAGSystem:
    """Simple system orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.discovery = DatabaseDiscovery(self.config)
        self.analyzer = SemanticAnalyzer(self.config)
        self.query_interface = QueryInterface(self.config)
        
        print("✅ Semantic Database RAG System initialized")
    
    async def run_database_discovery(self, limit: int = None):
        """Option 1: Database Discovery with view analysis"""
        print("🔍 DATABASE DISCOVERY")
        print("=" * 50)
        
        if limit:
            print(f"📊 Analyzing top {limit} priority objects")
        else:
            print("📊 Analyzing ALL database objects")
        
        success = await self.discovery.discover_database(limit)
        
        if success:
            tables = self.discovery.get_tables()
            table_count = sum(1 for t in tables if t.object_type in ['BASE TABLE', 'TABLE'])
            view_count = sum(1 for t in tables if t.object_type == 'VIEW')
            
            print(f"✅ Discovery completed!")
            print(f"   📊 Found {len(tables)} objects ({table_count} tables, {view_count} views)")
            print(f"   📝 Sample data: 5 rows per object")
            print(f"   🔗 Relationships: {len(self.discovery.get_relationships())} discovered")
            
            return True
        else:
            print("❌ Discovery failed")
            return False
    
    async def run_semantic_analysis(self):
        """Option 2: Semantic Analysis with business intelligence"""
        print("\n🧠 SEMANTIC ANALYSIS")
        print("=" * 50)
        
        # Load tables
        tables = self.discovery.get_tables()
        if not tables:
            if self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
                print(f"   📊 Loaded {len(tables)} tables from discovery cache")
            else:
                print("❌ No tables found. Run discovery first.")
                return False
        
        print(f"🧠 Analyzing {len(tables)} tables for business entities")
        
        success = await self.analyzer.analyze_tables(tables)
        
        if success:
            print("✅ Semantic analysis completed!")
            
            domain = self.analyzer.get_domain()
            if domain:
                print(f"   🏢 Domain: {domain.domain_type}")
                print(f"   🎯 Capabilities: {len([c for c in domain.capabilities.values() if c])} query types")
            
            return True
        else:
            print("❌ Semantic analysis failed")
            return False
    
    async def run_interactive_queries(self):
        """Option 3: 4-Stage Automated Query Pipeline"""
        print("\n💬 INTERACTIVE QUERIES - 4-Stage Pipeline")
        print("=" * 50)
        
        # Load data
        tables = self.analyzer.get_tables() or self.discovery.get_tables()
        
        if not tables:
            # Try loading from cache
            if self.analyzer.load_from_cache():
                tables = self.analyzer.get_tables()
            elif self.discovery.load_from_cache():
                tables = self.discovery.get_tables()
        
        if not tables:
            print("❌ No tables available. Please run:")
            print("   1. Database Discovery")
            print("   2. Semantic Analysis")
            return False
        
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        print(f"🚀 Starting 4-stage automated pipeline with {len(tables)} tables")
        if domain:
            print(f"   🏢 Domain: {domain.domain_type}")
        if relationships:
            print(f"   🔗 Relationships: {len(relationships)}")
        
        await self.query_interface.start_interactive_session(tables, domain, relationships)
        return True
    
    def show_system_status(self):
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("=" * 40)
        
        # Discovery status
        discovery_tables = self.discovery.get_tables()
        if discovery_tables:
            table_count = sum(1 for t in discovery_tables if t.object_type in ['BASE TABLE', 'TABLE'])
            view_count = sum(1 for t in discovery_tables if t.object_type == 'VIEW')
            print(f"📋 Discovery: {len(discovery_tables)} objects ({table_count} tables, {view_count} views)")
        else:
            print("📋 Discovery: Not completed")
        
        # Semantic analysis status  
        analyzer_tables = self.analyzer.get_tables()
        domain = self.analyzer.get_domain()
        relationships = self.analyzer.get_relationships()
        
        if analyzer_tables:
            classified_count = sum(1 for t in analyzer_tables if hasattr(t, 'entity_type') and t.entity_type != 'Unknown')
            print(f"🧠 Semantic: {len(analyzer_tables)} tables, {classified_count} classified")
        else:
            print("🧠 Semantic: Not completed")
        
        if relationships:
            print(f"🔗 Relationships: {len(relationships)} discovered")
        
        if domain:
            print(f"🏢 Domain: {domain.domain_type}")
            enabled_caps = [cap for cap, enabled in domain.capabilities.items() if enabled]
            if enabled_caps:
                print(f"🎯 Capabilities: {', '.join(cap.replace('_', ' ') for cap in enabled_caps)}")

def main():
    """Main application entry point"""
    print("🚀 SEMANTIC DATABASE RAG SYSTEM")
    print("Simple, Readable, and Maintainable")
    print("=" * 60)
    
    # Load environment
    load_environment()
    
    # Initialize system
    try:
        system = SemanticRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Check your .env file:")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - DATABASE_CONNECTION_STRING")
        return
    
    while True:
        print("\n" + "="*60)
        print("SEMANTIC DATABASE RAG SYSTEM:")
        print("1. 🔍 Database Discovery - Analyze database structure + views")
        print("2. 🧠 Semantic Analysis - Classify entities + relationships")  
        print("3. 💬 Interactive Queries - 4-stage automated pipeline")
        print("4. 📊 System Status - Show current state")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("Enter your choice (0-4): ").strip()
            
            if choice == '0':
                print("👋 Thanks for using the Semantic Database RAG System!")
                break
            elif choice == '1':
                asyncio.run(system.run_database_discovery())
            elif choice == '2':
                asyncio.run(system.run_semantic_analysis())
            elif choice == '3':
                asyncio.run(system.run_interactive_queries())
            elif choice == '4':
                system.show_system_status()
            else:
                print(f"❌ Invalid choice: '{choice}'. Please enter 0-4.")
        
        except KeyboardInterrupt:
            print("\n⏸️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()