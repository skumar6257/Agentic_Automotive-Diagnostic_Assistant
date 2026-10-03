import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
from neo4j import GraphDatabase
from infra.factory import factory

def ingest_dtc_graph():
    print(f"--- Starting DTC Graph Ingestion ({factory.deployment_mode} Mode) ---")

    # 1. Get Neo4j Credentials from Factory
    config = factory.get_graph_db_config()
    uri = config["uri"]
    user = config["username"]
    password = config["password"]

    if not uri:
        print("ERROR: Please set AURA_DB_URI (Cloud) or LOCAL_GRAPH_URI (Offline) in .env file.")

    # 2. Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    print(f"Connected to Neo4j: {uri}")

    #3. Read CSV and run Cypher queries
    csv_path = "data/raw/dtc_codes.csv"

    cypher_query = """
    MERGE (d:DTC {code: $code, description: $description})
    MERGE (p:Part {name: $part})
    MERGE (s:Subsystem {name: $subsystem})
    MERGE (d)-[:INDICATES_ISSUE_IN]->(p)
    MERGE (p)-[:LOCATED_IN]->(s)
    """

    try:
        with driver.session() as session:
            with open(csv_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                count = 0
                for row in reader:
                    session.run(cypher_query,
                        code=row['DTC'],
                        description=row['Description'],
                        part=row['Part'],
                        subsystem=row['Subsystem']
                        ) 
                    count += 1
            print("DTC Graph Ingestion Complete!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()   

if __name__ == "__main__":
    ingest_dtc_graph()