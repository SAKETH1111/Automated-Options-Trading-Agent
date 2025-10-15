#!/usr/bin/env python3
"""
Test script to verify S3 connection and credentials for Polygon flat files
"""

import os
import sys
import boto3
from loguru import logger

def test_s3_connection():
    """Test S3 connection with provided credentials"""
    logger.info("Testing S3 connection to Polygon flat files...")
    
    try:
        # Initialize S3 client with provided credentials
        s3_client = boto3.client(
            's3',
            endpoint_url='https://files.polygon.io',
            aws_access_key_id='88976b8e-f103-4d07-959f-4d13ce686dd2',
            aws_secret_access_key='wWrUjjcksqLDPntXbJb72kiFzAwyqIpY',
            region_name='us-east-1'
        )
        
        bucket_name = 'flatfiles'
        
        # Test 1: List bucket contents
        logger.info("1. Testing bucket access...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix="options/",
                Delimiter="/",
                MaxKeys=10
            )
            
            if 'CommonPrefixes' in response:
                logger.info(f"✅ Successfully connected to bucket '{bucket_name}'")
                logger.info(f"Found {len(response['CommonPrefixes'])} top-level prefixes:")
                for prefix in response['CommonPrefixes'][:5]:  # Show first 5
                    logger.info(f"  - {prefix['Prefix']}")
            else:
                logger.warning("⚠️ No prefixes found in bucket")
                
        except Exception as e:
            logger.error(f"❌ Error accessing bucket: {e}")
            return False
        
        # Test 2: List options data types
        logger.info("\n2. Testing options data types...")
        data_types = ['trades', 'quotes', 'aggregates']
        
        for data_type in data_types:
            try:
                prefix = f"options/{data_type}/"
                response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix,
                    Delimiter="/",
                    MaxKeys=5
                )
                
                if 'CommonPrefixes' in response:
                    logger.info(f"✅ Found {data_type} data: {len(response['CommonPrefixes'])} date folders")
                    # Show first few dates
                    for prefix_obj in response['CommonPrefixes'][:3]:
                        date_str = prefix_obj['Prefix'].replace(prefix, '').rstrip('/')
                        logger.info(f"  - {date_str}")
                else:
                    logger.warning(f"⚠️ No {data_type} data found")
                    
            except Exception as e:
                logger.error(f"❌ Error accessing {data_type} data: {e}")
        
        # Test 3: Try to list a specific date
        logger.info("\n3. Testing specific date access...")
        try:
            # Try to find the most recent date
            trades_prefix = "options/trades/"
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=trades_prefix,
                Delimiter="/",
                MaxKeys=1
            )
            
            if 'CommonPrefixes' in response:
                latest_date_prefix = response['CommonPrefixes'][0]['Prefix']
                date_str = latest_date_prefix.replace(trades_prefix, '').rstrip('/')
                logger.info(f"✅ Found latest trades date: {date_str}")
                
                # Try to list files for this date
                date_prefix = f"options/trades/{date_str}/"
                file_response = s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=date_prefix,
                    MaxKeys=5
                )
                
                if 'Contents' in file_response:
                    logger.info(f"✅ Found {len(file_response['Contents'])} files for {date_str}:")
                    for obj in file_response['Contents'][:3]:
                        file_name = obj['Key'].split('/')[-1]
                        file_size = obj['Size']
                        logger.info(f"  - {file_name} ({file_size} bytes)")
                else:
                    logger.warning(f"⚠️ No files found for {date_str}")
            else:
                logger.warning("⚠️ No date folders found")
                
        except Exception as e:
            logger.error(f"❌ Error accessing specific date: {e}")
        
        logger.info("\n✅ S3 connection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 connection test failed: {e}")
        return False

def test_flat_files_client():
    """Test the flat files client with updated credentials"""
    logger.info("\n" + "="*60)
    logger.info("Testing PolygonFlatFilesClient with updated credentials")
    logger.info("="*60)
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.market_data.polygon_flat_files import PolygonFlatFilesClient
        
        # Initialize client
        client = PolygonFlatFilesClient()
        logger.info("✅ PolygonFlatFilesClient initialized successfully")
        
        # Test listing available dates
        logger.info("\n1. Testing available dates listing...")
        
        for data_type in ['trades', 'quotes', 'aggregates']:
            dates = client.list_available_dates(data_type)
            if dates:
                logger.info(f"✅ Found {len(dates)} available {data_type} dates")
                logger.info(f"   Latest: {dates[-1]}")
                logger.info(f"   Earliest: {dates[0]}")
            else:
                logger.warning(f"⚠️ No {data_type} dates found")
        
        # Test downloading a small sample
        logger.info("\n2. Testing data download...")
        
        # Try to get the most recent date
        trades_dates = client.list_available_dates("trades")
        if trades_dates:
            test_date = trades_dates[-1]
            logger.info(f"Testing download for {test_date}...")
            
            # Try to download a small sample
            file_path = client.download_data("trades", test_date)
            if file_path and file_path.exists():
                logger.info(f"✅ Successfully downloaded trades data: {file_path}")
                logger.info(f"   File size: {file_path.stat().st_size} bytes")
            else:
                logger.warning("⚠️ Download failed or file not found")
        else:
            logger.warning("⚠️ No trades dates available for testing")
        
        logger.info("\n✅ Flat files client test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Flat files client test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting S3 Connection and Flat Files Testing")
    logger.info(f"Timestamp: {os.popen('date').read().strip()}")
    
    # Test S3 connection
    s3_success = test_s3_connection()
    
    # Test flat files client
    client_success = test_flat_files_client()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"S3 Connection: {'✅ PASSED' if s3_success else '❌ FAILED'}")
    logger.info(f"Flat Files Client: {'✅ PASSED' if client_success else '❌ FAILED'}")
    
    if s3_success and client_success:
        logger.info("\n🎉 All tests passed! S3 credentials are working correctly.")
        logger.info("Your trading agent can now access historical flat files data!")
        return True
    else:
        logger.error("\n❌ Some tests failed. Please check the credentials and configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
