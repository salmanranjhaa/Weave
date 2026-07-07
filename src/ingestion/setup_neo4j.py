"""
Neo4j Graph Ingestion Script
Constructs an intricate graph representing users, their teams, tickets, and blocked_by relationships.
"""
import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

class ERPGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def wipe_db(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def load_users(self, users):
        ceo_id = None
        for u in users:
            if str(u.get('title','')).lower() == 'ceo' or (u.get('manager_id') is None and u.get('role', '') == 'CEO'):
                ceo_id = u['user_id']
                break
        if not ceo_id and users:
            ceo_id = users[0]['user_id']
            
        with self.driver.session() as session:
            for u in users:
                session.run('''
                    MERGE (dept:Department {name: $department})
                    MERGE (access:AccessLevel {level: $access_level})
                    CREATE (p:Person {
                        user_id: $user_id, name: $name, role: $role, salary: $salary
                    })
                    CREATE (p)-[:WORKS_IN]->(dept)
                    CREATE (p)-[:HAS_ACCESS]->(access)
                ''', user_id=u['user_id'], name=u['name'], role=u['role'], 
                     department=u.get('department', 'General'), 
                     access_level=u.get('access_level', 1),
                     salary=u.get('salary', 0))
                     
            # Create manager relationships
            for u in users:
                manager = u.get('manager_id')
                if not manager and u['user_id'] != ceo_id:
                    manager = ceo_id
                if manager:
                    session.run('''
                        MATCH (emp:Person {user_id: $user_id})
                        MATCH (mgr:Person {user_id: $manager_id})
                        CREATE (emp)-[:REPORTS_TO]->(mgr)
                    ''', user_id=u['user_id'], manager_id=manager)

    def load_tickets(self, tickets):
        valid_tickets = [t for t in tickets if isinstance(t, dict)]
        with self.driver.session() as session:
            for t in valid_tickets:
                session.run('''
                    MERGE (sprint:Sprint {name: coalesce($sprint, "Backlog")})
                    MERGE (status:Status {name: coalesce($status, "Open")})
                    MERGE (customer:Customer {name: coalesce($customer_account, "Internal")})
                    CREATE (t:Ticket {
                        ticket_id: $ticket_id, title: $title, type: $type, priority: $priority
                    })
                    CREATE (t)-[:ASSIGNED_TO_SPRINT]->(sprint)
                    CREATE (t)-[:CURRENT_STATUS]->(status)
                    CREATE (t)-[:AFFECTS_CUSTOMER]->(customer)
                ''', ticket_id=t.get('ticket_id'), title=t.get('title'), type=t.get('type'),
                     status=t.get('status'), priority=t.get('priority'), sprint=t.get('sprint'),
                     customer_account=t.get('customer_account'))
                
                if t.get('assignee_id'):
                    session.run('''
                        MATCH (p:Person {user_id: $assignee_id})
                        MATCH (t:Ticket {ticket_id: $ticket_id})
                        CREATE (p)-[:ASSIGNED_TO]->(t)
                    ''', assignee_id=t['assignee_id'], ticket_id=t['ticket_id'])
                if t.get('reporter_id'):
                    session.run('''
                        MATCH (p:Person {user_id: $reporter_id})
                        MATCH (t:Ticket {ticket_id: $ticket_id})
                        CREATE (p)-[:REPORTED]->(t)
                    ''', reporter_id=t['reporter_id'], ticket_id=t['ticket_id'])
            
            # Create Blocker relationships
            for t in valid_tickets:
                blocked_by = t.get('blocked_by', [])
                for blocker_id in blocked_by:
                    session.run('''
                        MATCH (blocked:Ticket {ticket_id: $ticket_id})
                        MATCH (blocker:Ticket {ticket_id: $blocker_id})
                        CREATE (blocked)-[:BLOCKED_BY]->(blocker)
                        CREATE (blocker)-[:BLOCKS]->(blocked)
                    ''', ticket_id=t['ticket_id'], blocker_id=blocker_id)

def main():
    print("Initializing Neo4j Graph Database...")
    try:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_pw = os.getenv("NEO4J_PASSWORD", "adminpassword")
        graph = ERPGraph(neo4j_uri, neo4j_user, neo4j_pw)
        graph.wipe_db()
        print("Wiped existing graph.")
        
        with open(os.path.join(DATA_DIR, 'users.json'), 'r') as f:
            users = json.load(f)
        
        print("Ingesting Organizational Hierarchy...")
        graph.load_users(users)
        
        with open(os.path.join(DATA_DIR, 'tickets.json'), 'r') as f:
            tickets = json.load(f)
            
        print("Ingesting Agile Tickets and Blocker chains...")
        graph.load_tickets(tickets)
        
        graph.close()
        print("Neo4j Ingestion Complete! Visualizations available at http://localhost:7474")
    except Exception as e:
        print(f"Error connecting to Neo4j. Make sure Docker container 'erp-neo4j' is running. {e}")

if __name__ == "__main__":
    main()
