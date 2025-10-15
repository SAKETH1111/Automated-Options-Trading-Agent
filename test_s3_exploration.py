#!/usr/bin/env python3
"""
Exploratory test script to discover the actual S3 bucket structure for Polygon flat files
"""

import boto3
from loguru import logger

def explore_s3_bucket():
    """Explore the S3 bucket structure to understand the correct paths"""
    logger.info("Exploring S3 bucket structure...")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url='https://files.polygon.io',
            aws_access_key_id='88976b8e-f103-4d07-959f-4d13ce686dd2',
            aws_secret_access_key='wWrUjjcksqLDPntXbJb72kiFzAwyqIpY',
            region_name='us-east-1'
        )
        
        bucket_name = 'flatfiles'
        
        # Test 1: List all objects in the bucket (with limit)
        logger.info("1. Listing all objects in bucket (first 20)...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=20
            )
            
            if 'Contents' in response:
                logger.info(f"✅ Found {len(response['Contents'])} objects in bucket:")
                for obj in response['Contents']:
                    logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
            else:
                logger.info("No objects found in bucket")
                
        except Exception as e:
            logger.error(f"❌ Error listing objects: {e}")
        
        # Test 2: Try different common prefixes
        logger.info("\n2. Testing common prefixes...")
        common_prefixes = [
            "",
            "options/",
            "polygon-options/",
            "data/",
            "flatfiles/",
            "trades/",
            "quotes/",
            "aggregates/"
        ]
        
        for prefix in common_prefixes:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix,
                    Delimiter="/",
                    MaxKeys=10
                )
                
                if 'CommonPrefixes' in response and response['CommonPrefixes']:
                    logger.info(f"✅ Found prefixes under '{prefix}':")
                    for prefix_obj in response['CommonPrefixes'][:5]:
                        logger.info(f"  - {prefix_obj['Prefix']}")
                elif 'Contents' in response and response['Contents']:
                    logger.info(f"✅ Found objects under '{prefix}':")
                    for obj in response['Contents'][:3]:
                        logger.info(f"  - {obj['Key']}")
                else:
                    logger.info(f"❌ No content under '{prefix}'")
                    
            except Exception as e:
                logger.error(f"❌ Error with prefix '{prefix}': {e}")
        
        # Test 3: Try to find any files with common extensions
        logger.info("\n3. Searching for files with common extensions...")
        extensions = ['.csv', '.gz', '.csv.gz', '.json', '.txt']
        
        for ext in extensions:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=10
                )
                
                if 'Contents' in response:
                    matching_files = [obj for obj in response['Contents'] if obj['Key'].endswith(ext)]
                    if matching_files:
                        logger.info(f"✅ Found {len(matching_files)} files with extension '{ext}':")
                        for obj in matching_files[:3]:
                            logger.info(f"  - {obj['Key']}")
                    else:
                        logger.info(f"❌ No files found with extension '{ext}'")
                        
            except Exception as e:
                logger.error(f"❌ Error searching for '{ext}': {e}")
        
        # Test 4: Try different bucket names
        logger.info("\n4. Testing different bucket names...")
        possible_buckets = [
            'flatfile',
            'flatfiles',
            'polygon-flatfiles',
            'polygon-flatfile',
            'polygon-options',
            'options-data'
        ]
        
        for test_bucket in possible_buckets:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=test_bucket,
                    MaxKeys=5
                )
                
                if 'Contents' in response:
                    logger.info(f"✅ Bucket '{test_bucket}' exists and has content!")
                    for obj in response['Contents'][:3]:
                        logger.info(f"  - {obj['Key']}")
                else:
                    logger.info(f"❌ Bucket '{test_bucket}' is empty or doesn't exist")
                    
            except Exception as e:
                logger.info(f"❌ Bucket '{test_bucket}' error: {e}")
        
        # Test 5: Try to get bucket location and properties
        logger.info("\n5. Getting bucket information...")
        try:
            location_response = s3_client.get_bucket_location(Bucket=bucket_name)
            logger.info(f"✅ Bucket location: {location_response.get('LocationConstraint', 'us-east-1')}")
        except Exception as e:
            logger.error(f"❌ Error getting bucket location: {e}")
        
        # Test 6: Try to list objects with different delimiters
        logger.info("\n6. Testing different delimiters...")
        delimiters = ['/', '_', '-', '.']
        
        for delimiter in delimiters:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Delimiter=delimiter,
                    MaxKeys=10
                )
                
                if 'CommonPrefixes' in response and response['CommonPrefixes']:
                    logger.info(f"✅ Found prefixes with delimiter '{delimiter}':")
                    for prefix_obj in response['CommonPrefixes'][:3]:
                        logger.info(f"  - {prefix_obj['Prefix']}")
                else:
                    logger.info(f"❌ No prefixes found with delimiter '{delimiter}'")
                    
            except Exception as e:
                logger.error(f"❌ Error with delimiter '{delimiter}': {e}")
        
        # Test 7: Search for specific patterns
        logger.info("\n7. Searching for specific patterns...")
        patterns = ['2024', '2023', 'trades', 'quotes', 'aggregates', 'options']
        
        for pattern in patterns:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=20
                )
                
                if 'Contents' in response:
                    matching_files = [obj for obj in response['Contents'] if pattern in obj['Key']]
                    if matching_files:
                        logger.info(f"✅ Found {len(matching_files)} files containing '{pattern}':")
                        for obj in matching_files[:3]:
                            logger.info(f"  - {obj['Key']}")
                    else:
                        logger.info(f"❌ No files found containing '{pattern}'")
                        
            except Exception as e:
                logger.error(f"❌ Error searching for '{pattern}': {e}")
        
        logger.info("\n✅ S3 exploration completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 exploration failed: {e}")
        return False

def main():
    """Main exploration function"""
    logger.info("🔍 Starting S3 Bucket Structure Exploration")
    logger.info("This will help us understand the correct path structure for Polygon flat files")
    
    success = explore_s3_bucket()
    
    if success:
        logger.info("\n🎉 Exploration completed! Check the output above to understand the bucket structure.")
        logger.info("Use this information to update the flat files client with the correct paths.")
    else:
        logger.error("\n❌ Exploration failed. Please check the credentials and try again.")

if __name__ == "__main__":
    main()
