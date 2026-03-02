from opensearch_service import OpenSearchService

def setup():
    os_service = OpenSearchService()
    if os_service.health_check():
        print("Connected to OpenSearch!")
        if os_service.create_index():
            print("Index 'arxiv-papers' created successfully.")
        else:
            print("Index already exists.")
    else:
        print("Could not connect to OpenSearch.")

if __name__ == "__main__":
    setup()