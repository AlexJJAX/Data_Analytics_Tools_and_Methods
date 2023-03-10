# PSQL to Python

## What the code does ?

This code snippet defines a function query_database that queries a PostgreSQL database using the provided database_uri and query and returns the result as a Pandas DataFrame.

To do this, the code first imports the necessary packages, including pandas, numpy, and sqlalchemy. It also sets up the database_uri with the appropriate credentials and port number.

Then, the function query_database is defined with two parameters: database_uri and query. Within the function, it uses sqlalchemy to create a connection to the database specified by the database_uri and then uses that connection to execute the provided query. The results are then stored in a Pandas DataFrame called "df".

After that, the function refreshes the query result to get an up-to-date version and closes the database connection. Finally, the function returns the "df" DataFrame.

Finally, the code calls the query_database function with the provided database_uri and query, and the resulting DataFrame is assigned to the variable df.

