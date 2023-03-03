import json
import sqlite3
from datetime import datetime

import requests
from bs4 import BeautifulSoup


class JobScraper:
    """
    A class for scraping job data from the Platsbanken API and storing it in a SQLite database.

    Attributes:
        api_root (str): The root URL for the Platsbanken API.
        text (str): The text content of the job posting.
        data (dict): The JSON data for a job posting.
        query (str): The search query for job postings.
        harvested_ids (list): A list of job IDs that have been harvested from the API.

    Author: https://github.com/lindgrenar/
    Repository: https://github.com/lindgrenar/yummydata
    """

    def __init__(self, query):
        """
        Initializes a new instance of the JobScraper class.

        Args:
            query (str): The search query for job postings.
        """
        self.api_root = "https://platsbanken-api.arbetsformedlingen.se/jobs/v1/"
        self.text = None
        self.data = None
        self.query = query
        self.harvested_ids = []

    def jobid_harvester(self):
        """
        Harvests job IDs from the Platsbanken API based on a search query.
        """
        url = self.api_root + "search/"
        headers = {'Content-Type': 'application/json'}

        for source in ["pb"]:
            data = dict(filters=[{"type": "freetext", "value": self.query}], fromDate=None, order="relevance",
                        maxRecords=25, startIndex=0, toDate=datetime.utcnow().isoformat() + 'Z',
                        source=source)

            while True:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                ads = response.json()['ads']
                if not ads:
                    break
                for ad in ads:
                    self.harvested_ids.append(ad['id'])
                data['startIndex'] += 25

    def database_commit(self):
        """
        Commits the harvested job IDs to a SQLite database.
        """
        try:
            conn = sqlite3.connect('jobid.db')
            c = conn.cursor()
            # Create a table if it doesn't exist
            c.execute(
                '''CREATE TABLE IF NOT EXISTS jobids (ID INTEGER PRIMARY KEY, rawtext TEXT, company TEXT, title TEXT, preprocessed TEXT)''')

            for jobid in self.harvested_ids:
                # Insert ID into the table if it doesn't already exist
                try:
                    c.execute("INSERT OR IGNORE INTO jobids (ID) VALUES (?)", (jobid,))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Error: Failed to execute SQL statement: {e}")

        except sqlite3.Error as e:
            print(f"Error: Failed to connect to database or execute SQL statement: {e}")

        finally:
            conn.close()

    def fetch_data_platsbanken(self):
        """
        Fetches job data from the Platsbanken API and stores it in a SQLite database.
        """
        try:
            conn = sqlite3.connect('jobid.db')
            c = conn.cursor()
            # Create a table if it doesn't exist
            c.execute(
                '''CREATE TABLE IF NOT EXISTS jobids (ID INTEGER PRIMARY KEY, rawtext TEXT, company TEXT, title TEXT, preprocessed TEXT)''')
            # Select IDs from the table where rawtext is NULL
            c.execute("SELECT ID FROM jobids WHERE rawtext IS NULL")
            ids_to_process = c.fetchall()

            total_ids = len(ids_to_process)

            for index, id_to_process in enumerate(ids_to_process):
                jobid = id_to_process[0]
                url = self.api_root + "job/" + str(jobid)
                headers = {'Accept': 'application/json'}
                response = requests.get(url, headers=headers)

                if not response.ok:
                    print(f"Error: Request failed with status code {response.status_code}")
                    continue

                print(f"Processing ID {index + 1} of {total_ids}. Response status code: {response.status_code}")

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    print("Error: Failed to decode JSON response")
                    continue

                title = data.get('title')
                company = data['company']['name']
                description = data.get('description')
                soup = BeautifulSoup(description, "html.parser")
                text = soup.get_text()
                # Update the table with the fetched data
                try:
                    c.execute("UPDATE jobids SET rawtext=?, company=?, title=? WHERE ID=?",
                              (text, company, title, jobid))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Error: Failed to execute SQL statement: {e}")

        except sqlite3.Error as e:
            print(f"Error: Failed to connect to database or execute SQL statement: {e}")
        finally:
            conn.close()

    def run(self):
        """
        Run
        """
        self.jobid_harvester()
        self.database_commit()
        self.fetch_data_platsbanken()
