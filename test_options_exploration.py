#!/usr/bin/env python3
"""
Targeted exploration script to find options data in the flatfiles bucket
"""

import boto3
from loguru import logger

def explore_options_data():
    """Explore the flatfiles bucket specifically for options data"""
    logger.info("Exploring flatfiles bucket for options data...")
    
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
        
        # Test 1: List all top-level prefixes
        logger.info("1. Listing all top-level prefixes in flatfiles bucket...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Delimiter="/",
                MaxKeys=50
            )
            
            if 'CommonPrefixes' in response:
                logger.info(f"✅ Found {len(response['CommonPrefixes'])} top-level prefixes:")
                for prefix in response['CommonPrefixes']:
                    logger.info(f"  - {prefix['Prefix']}")
            else:
                logger.info("No prefixes found")
                
        except Exception as e:
            logger.error(f"❌ Error listing prefixes: {e}")
        
        # Test 2: Look for options-specific prefixes
        logger.info("\n2. Searching for options-related prefixes...")
        options_keywords = ['options', 'option', 'opts', 'equity_options', 'stock_options']
        
        for keyword in options_keywords:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=keyword,
                    Delimiter="/",
                    MaxKeys=20
                )
                
                if 'CommonPrefixes' in response and response['CommonPrefixes']:
                    logger.info(f"✅ Found {keyword} data:")
                    for prefix_obj in response['CommonPrefixes'][:5]:
                        logger.info(f"  - {prefix_obj['Prefix']}")
                else:
                    logger.info(f"❌ No {keyword} data found")
                    
            except Exception as e:
                logger.error(f"❌ Error searching for {keyword}: {e}")
        
        # Test 3: Search for files containing 'options' in the name
        logger.info("\n3. Searching for files containing 'options'...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=100
            )
            
            if 'Contents' in response:
                options_files = [obj for obj in response['Contents'] if 'options' in obj['Key'].lower()]
                if options_files:
                    logger.info(f"✅ Found {len(options_files)} files containing 'options':")
                    for obj in options_files[:10]:
                        logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                else:
                    logger.info("❌ No files containing 'options' found")
            else:
                logger.info("❌ No files found in bucket")
                
        except Exception as e:
            logger.error(f"❌ Error searching for options files: {e}")
        
        # Test 4: Look for common data types
        logger.info("\n4. Searching for common data types...")
        data_types = ['trades', 'quotes', 'aggregates', 'bars', 'ticks']
        
        for data_type in data_types:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=50
                )
                
                if 'Contents' in response:
                    matching_files = [obj for obj in response['Contents'] if data_type in obj['Key'].lower()]
                    if matching_files:
                        logger.info(f"✅ Found {len(matching_files)} files containing '{data_type}':")
                        for obj in matching_files[:5]:
                            logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                    else:
                        logger.info(f"❌ No files containing '{data_type}' found")
                        
            except Exception as e:
                logger.error(f"❌ Error searching for {data_type}: {e}")
        
        # Test 5: Look for recent dates (2024, 2025)
        logger.info("\n5. Searching for recent data...")
        recent_years = ['2024', '2025']
        
        for year in recent_years:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=50
                )
                
                if 'Contents' in response:
                    year_files = [obj for obj in response['Contents'] if year in obj['Key']]
                    if year_files:
                        logger.info(f"✅ Found {len(year_files)} files from {year}:")
                        for obj in year_files[:5]:
                            logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                    else:
                        logger.info(f"❌ No files from {year} found")
                        
            except Exception as e:
                logger.error(f"❌ Error searching for {year}: {e}")
        
        # Test 6: Look for specific file patterns
        logger.info("\n6. Searching for specific file patterns...")
        patterns = ['.csv.gz', '.csv', 'trades_', 'quotes_', 'aggregates_']
        
        for pattern in patterns:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    MaxKeys=50
                )
                
                if 'Contents' in response:
                    pattern_files = [obj for obj in response['Contents'] if pattern in obj['Key']]
                    if pattern_files:
                        logger.info(f"✅ Found {len(pattern_files)} files with pattern '{pattern}':")
                        for obj in pattern_files[:5]:
                            logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                    else:
                        logger.info(f"❌ No files with pattern '{pattern}' found")
                        
            except Exception as e:
                logger.error(f"❌ Error searching for pattern '{pattern}': {e}")
        
        logger.info("\n✅ Options data exploration completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Options exploration failed: {e}")
        return False

def main():
    """Main exploration function"""
    logger.info("🔍 Starting Options Data Exploration in flatfiles bucket")
    logger.info("This will help us find the correct path structure for options data")
    
    success = explore_options_data()
    
    if success:
        logger.info("\n🎉 Exploration completed! Check the output above to find options data.")
        logger.info("Use this information to update the flat files client with the correct paths.")
    else:
        logger.error("\n❌ Exploration failed. Please check the credentials and try again.")

if __name__ == "__main__":
    main()
