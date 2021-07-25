from workspace_utils import active_session
 
with active_session():
    # do long-running work here
    import configparser
    from datetime import datetime
    import os
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.functions import (year, month, dayofmonth, hour,
                                   weekofyear, dayofweek, date_format,
                                   monotonically_increasing_id)
    from pyspark.sql import functions as F
    from pyspark.sql import functions as T


    config = configparser.ConfigParser()
    config.read('dl.cfg')

    os.environ['AWS_ACCESS_KEY_ID']=config['KEYS']['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY']=config['KEYS']['AWS_SECRET_ACCESS_KEY']


    def create_spark_session():
        """
        Creates an Apache Spark session.
        """
        spark = SparkSession \
            .builder \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
            .getOrCreate()
        return spark


    def process_song_data(spark, input_data, output_data):
        """
        Processes the raw song data and stores output 'song' and 'artists' tables as parquet files in AWS S3
        
        Inputs:
        - spark (SparkSession): Spark session object
        - input_data (string): String with path to where all the input data is stored
        - output_data (string): String with path to where all the output data should be written to
        
        Returns:
        None
        """
        
        # get filepath to song data file
        song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

        # read song data file
        df = spark.read.json(song_data)

        # extract columns to create songs table
        songs_table = df.select(df['song_id'], df['title'],df['artist_id'],df['year'],df['duration']).dropDuplicates(subset=['song_id'])

        # write songs table to parquet files partitioned by year and artist
        songs_table.write.mode('overwrite').partitionBy('year', 'artist_id').parquet(os.path.join(output_data, 'songs', 'songs_table.parquet'))

        # extract columns to create artists table
        artists_table = df.select(df['artist_id'],df['artist_name'],df['artist_location'],df['artist_latitude'],df['artist_longitude']).dropDuplicates(subset=['artist_id'])

        # write artists table to parquet files
        artists_table.write.mode('overwrite').parquet(os.path.join(output_data, 'artists','artists_table.parquet'))


    def process_log_data(spark, input_data, output_data):
        
        """
        Processes the raw log data and stores output 'users', 'time' and 'songplays' tables as parquet files in AWS S3
        
        Inputs:
        - spark (SparkSession): Spark session object
        - input_data (string): String with path to where all the input data is stored
        - output_data (string): String with path to where all the output data should be written to
        
        Returns:
        None
        """
        # get filepath to log data file
        log_data = os.path.join(input_data, 'log_data/*.json')

        # read log data file
        df = spark.read.json(log_data)

        # filter by actions for song plays
        actions_df = df.filter(df.page == 'NextSong').select('ts','userId','level','song','artist','sessionId','location','userAgent')

        # extract columns for users table    
        users_table = df.select(df['userId'],df['firstName'],df['lastName'],df['gender'],df['level']).dropDuplicates(subset=['user_id'])

        # write users table to parquet files
        users_table.write.mode('overwrite').parquet(os.path.join(output_data,'users', 'users_table.parquet'))

        # create timestamp column from original timestamp column
        get_timestamp = udf(lambda x: str(int(int(x)/ 1000)))
        actions_df = actions_df.withColumn("timestamp", get_timestamp(actions_df.ts))

        # create datetime column from original timestamp column
        get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x)/ 1000)))
        actions_df = actions_df.withColumn("datetime", get_datetime(actions_df.ts))

        # extract columns to create time table
        time_table = actions_df.select("datetime")\
                                .withColumn("start_time", actions_df.datetime)\
                                .withColumn("hour", hour("datetime"))\
                                .withColumn("day", dayofmonth("datetime"))\
                                .withColumn("week", weekofyear("datetime"))\
                                .withColumn("month", month("datetime"))\
                                .withColumn("year", year("datetime"))\
                                .withColumn("weekday", dayofweek("datetime"))\
                                .dropDuplicates(subset=['start_time'])

        # write time table to parquet files partitioned by year and month
        time_table.write.mode('overwrite').partitionBy("year","month").parquet(os.path.join(output_data,'time', 'time_table.parquet'))

        # read in song data to use for songplays table
        song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
        song_df = spark.read.json(song_data)
        
        # extract columns from joined song and log datasets to create songplays table 
        actions_df = actions_df.alias('log_df')
        song_df = song_df.alias('song_df')
        joined_df = actions_df.join(song_df, col('log_df.artist') == col('song_df.artist_name'),'inner')
        songplays_table = joined_df.select(
            col('log_df.datetime').alias('start_time'),
            col('log_df.userId').alias('user_id'),
            col('log_df.level').alias('level'),
            col('song_df.song_id').alias('song_id'),
            col('song_df.artist_id').alias('artist_id'),
            col('log_df.sessionId').alias('session_id'),
            col('log_df.location').alias('location'), 
            col('log_df.userAgent').alias('user_agent'),
            year('log_df.datetime').alias('year'),
            month('log_df.datetime').alias('month')) \
            .withColumn('songplay_id', monotonically_increasing_id()).dropDuplicates(subset=['song_id','start_time','artist_id'])

        # write songplays table to parquet files partitioned by year and month
        songplays_table.write.mode('overwrite').partitionBy('year','month').parquet(os.path.join(output_data,'songplays', 'songplays_table.parquet'))


    def main():
        """
        Performs the following operations:
        - Get or create spark session
        - Read song and log data from S3
        - Transform data
        - Write data into parquet files on S3
        """
        
        spark = create_spark_session()
        input_data = "s3a://udacity-dend"
        output_data = "s3a://udacity-data-lake-proj"

        process_song_data(spark, input_data, output_data)    
        process_log_data(spark, input_data, output_data)


    if __name__ == "__main__":
        main()
