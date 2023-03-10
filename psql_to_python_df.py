import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------POSTGRESQL SESSION CONNECTION BLOCK--------------------------

database_uri = 'postgresql://username:password@localhost:local port number/name of the database'

# postgresql:// indicates that the connection is to a PostgreSQL database.
# [username] is the username that will be used to connect to the database.
# [password] is the password that will be used to authenticate the user.
# [local port number] is the port number of the PostgreSQL server.
# [name of the database] is the name of the database that will be connected to.

query = 'SELECT * FROM customers'

def query_database(database_uri, query):
    # Set up the database connection and start a session
    engine = sqlalchemy.create_engine(database_uri)
    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()

    # Query the database and store the result in a Pandas DataFrame
    df = pd.read_sql(query, con=engine)

    # Refresh the query result to get an up-to-date version
    for instance in session:
        session.refresh()

    # Close the session and the database connection
    session.close()
    engine.dispose()

    # Return the DataFrame
    return df

df = query_database(database_uri, query)
