from datetime import datetime
import json
import sqlite3

import numpy as np


class DatabaseWriter:
    def __init__(self, config):
        """
        Initializes the DatabaseWriter object, setting up the database connection.

        :param config: Configuration dictionary containing 'output_db' as the path to the SQLite database.
        """
        self.db_path = config.output_db
        # Set the path for the SQLite database
        self.connect_db()  # Connect to the SQLite database

    def connect_db(self):
        """
        Connects to the SQLite database, creating it if it doesn't exist.
        """
        self.conn = sqlite3.connect(self.db_path)
        # Check if the 'analytics' table exists
        self.cursor = self.conn.cursor()

        # Drop the existing 'analytics' table if it exists
        self.cursor.execute("DROP TABLE IF EXISTS analytics")

        # Create a new 'analytics' table with the desired schema
        self.cursor.execute(
            """CREATE TABLE analytics (
                                frame INTEGER,
                                x INTEGER,
                                y INTEGER,
                                team INTEGER,
                                id INTEGER,
                                cls INTEGER,
                                conf REAL)"""
        )

        self.conn.commit()

    def update_db(self, analytics_dict, scores_dict=None):
        """
        Updates the SQLite database by inserting a new row with the latest data.
        """

        # Check and convert only if the object is a NumPy array
        def convert_to_list(obj):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj

        # Extract data for each column, ensuring any NumPy arrays are converted to lists
        # current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql_cols = analytics_dict.loc[
            :, ["frame", "x_anchor", "y_anchor", "team", "id", "cls", "conf"]
        ]
        # Assume frame_number is calculated or obtained elsewhere
        # frame_number = analytics_dict['frame']  # Example to derive frame_number, adjust as necessary
        # Get the goal counts for each team from the scores_dict
        # goals_team_a = scores_dict.get('a', 0)
        # goals_team_b = scores_dict.get('b', 0)
        # Insert the data into the database
        self.cursor.executemany(
            """INSERT INTO analytics (frame, x, y, team, id, cls, conf)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            sql_cols.values,
        )
        self.conn.commit()

    def close_db(self):
        """
        Closes the database connection.
        """
        self.conn.close()
