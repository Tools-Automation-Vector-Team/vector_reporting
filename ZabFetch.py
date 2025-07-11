from typing import Optional, Dict, Any, List, Union
import mysql.connector
import psycopg2
from mysql.connector import Error as MySQLError
from psycopg2 import Error as PostgresError
from datetime import datetime, timezone
import re
import statistics
import math
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import time
import json
import os


# Configure logging with rotating file handler
log_dir = '/var/logs/vector'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'vector_lib_db.log')

handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

class ZabbixDBError(Exception):
    """Custom exception for ZabbixDB errors."""
    pass

class QueryBuilder:
    """Helper class to build and validate SQL queries."""
    
    @staticmethod
    def build_host_status_query() -> str:
        return """
        SELECT h.hostid, h.host, h.status
        FROM hosts h
        WHERE h.host = %s AND h.flags IN (0, 4)
        """

    @staticmethod
    def build_item_detail_query(itemid: int = None,item_name:str=None,hostname: str = None) -> str:
        base_query = """
        SELECT i.itemid, i.hostid, i.name, i.history, i.trends, i.value_type,i.delay,i.master_itemid,
            i.status, i.units, h.host
        FROM items i
        JOIN hosts h ON i.hostid = h.hostid
        """

        conditions = []
        
        if itemid:
            conditions.append(f"i.itemid = {itemid}")
        if item_name:
            conditions.append(f"i.name = '{item_name}'")
        if hostname:
            conditions.append(f"h.host = '{hostname}'")            
        if conditions:
            conditions.append("h.status = 0 AND h.flags IN (0, 4)")
            return base_query + "WHERE " + " AND ".join(conditions)
        
        return base_query

    @staticmethod
    def build_trend_query(table_name: str, statistical_measure: str) -> str:
        valid_tables = {'trends', 'trends_uint'}
        if table_name not in valid_tables:
            raise ZabbixDBError(f"Invalid trend table: {table_name}")
        
        value = 'value_avg as value'
        if statistical_measure == 'all':
            value = 'num, value_avg, value_min, value_max'
        elif statistical_measure in ['min', 'max']:
            value = f"value_{statistical_measure} as value"
            
        return f"""
        SELECT clock, {value}
        FROM {table_name}
        WHERE itemid = %s
        AND clock BETWEEN %s AND %s
        ORDER BY clock DESC
        """

    @staticmethod
    def build_history_query(table_name: str) -> str:
        valid_tables = {'history', 'history_str', 'history_log', 'history_uint', 'history_text'}
        if table_name not in valid_tables:
            raise ZabbixDBError(f"Invalid history table: {table_name}")
        return f"""
        SELECT clock, value
        FROM {table_name}
        WHERE itemid = %s
        AND clock BETWEEN %s AND %s
        ORDER BY clock DESC
        """
    
    @staticmethod
    def build_host_by_tag_query(hostname: str=None, tag_name: str = None,tag_value: str = None) -> str:
        base_query = f"""
        SELECT ht.tag, ht.value,h.host
        FROM hosts h
        JOIN hosts_templates hst ON h.hostid = hst.hostid
        JOIN host_tag ht ON ht.hostid IN (h.hostid, hst.templateid)
        """
        conditions = []

        if hostname:
            conditions.append(f"h.host = '{hostname}'")

        if tag_name:
            conditions.append(f"ht.tag = '{tag_name}'")

        if tag_value:
            conditions.append(f"ht.value = '{tag_value}'")

        if conditions:
            return base_query + " WHERE " + " AND ".join(conditions)

        return base_query
    
    @staticmethod
    def build_item_by_tag_query(item_name: str = None, hostname: str = None, tag_name: str = None, tag_value: str = None) -> str:
        values = "it.tag, it.value, i.name" if not hostname else "it.tag, it.value, i.name, h.host"
        base_query = f"""
        SELECT DISTINCT {values}
        from item_tag it
        JOIN items i ON i.itemid = it.itemid
        JOIN hosts h ON i.hostid = h.hostid
        """
        conditions = []
        if item_name:
            conditions.append(f"i.name = '{item_name}'")
        if hostname:
            conditions.append(f"h.host = '{hostname}'")

        if tag_name:
            conditions.append(f"it.tag = '{tag_name}'")

        if tag_value:
            conditions.append(f"it.value = '{tag_value}'")

        if conditions:
            return base_query + " WHERE " + " AND ".join(conditions)

        return base_query
    
    @staticmethod
    def build_tag_for_problem_query(eventid: int = None,event_name: str = None, tag_name: str = None, tag_value: str = None) -> str:
        base_query = """
        SELECT et.tag, et.value, et.eventid,e.name
        FROM event_tag et
        JOIN events e ON et.eventid = e.eventid
        """
        conditions = []
        if eventid:
            conditions.append(f"et.eventid = '{eventid}'")
        if event_name:
            conditions.append(f"e.name = '{event_name}'")

        if tag_name:
            conditions.append(f"et.tag = '{tag_name}'")

        if tag_value:
            conditions.append(f"et.value = '{tag_value}'")

        if conditions:
            return base_query + " WHERE " + " AND ".join(conditions)

        return base_query

    @staticmethod
    def build_macro_host_query(hostname: str, macro_name: str = None) -> str:
        base_query = f"""
        WITH selected_host AS (
            SELECT hostid, host
            FROM hosts
            WHERE host = '{hostname}'
        ),
        host_macros AS (
            SELECT macro, value
            FROM hostmacro
            WHERE hostid = (SELECT hostid FROM selected_host)
        ),
        template_macros AS (
            SELECT ht.templateid, hm.macro, hm.value
            FROM hosts_templates ht
            JOIN hostmacro hm ON hm.hostid = ht.templateid
            WHERE ht.hostid = (SELECT hostid FROM selected_host)
        )
        SELECT
            hm.macro,
            hm.value,
            sh.host,
            sh.hostid
        FROM
            selected_host sh
        JOIN
            host_macros hm ON 1=1

        UNION ALL

        SELECT
            tm.macro,
            tm.value,
            sh.host,
            sh.hostid
        FROM
            selected_host sh
        JOIN
            template_macros tm
        LEFT JOIN
            host_macros hm ON hm.macro = tm.macro
        WHERE
            hm.macro IS NULL

        """

        if macro_name:
            base_query = f"""
            WITH selected_host AS (
                SELECT hostid, host
                FROM hosts
                WHERE host = 'Zabbix server'
            ),
            host_macros AS (
                SELECT macro, value
                FROM hostmacro
                WHERE hostid = (SELECT hostid FROM selected_host)
                AND macro = '{macro_name}'
            ),
            template_macros AS (
                SELECT ht.templateid, hm.macro, hm.value
                FROM hosts_templates ht
                JOIN hostmacro hm ON hm.hostid = ht.templateid
                WHERE ht.hostid = (SELECT hostid FROM selected_host) AND hm.macro = '{macro_name}'
            )
            SELECT
                hm.macro,
                hm.value,
                sh.host,
                sh.hostid
            FROM
                selected_host sh
            JOIN
                host_macros hm ON 1=1

            UNION ALL

            SELECT
                tm.macro,
                tm.value,
                sh.host,
                sh.hostid
            FROM
                selected_host sh
            JOIN
                template_macros tm
            LEFT JOIN
                host_macros hm ON hm.macro = tm.macro
            WHERE
                hm.macro IS NULL;
                        """



        return base_query
    
    @staticmethod
    def build_host_by_group_query(hostname: str = None,host_group: str =None) -> str:
        
        base_query = """
            SELECT g.name AS group_name, h.host AS host_name
            FROM hstgrp g
            JOIN hosts_groups hg ON g.groupid = hg.groupid
            JOIN hosts h ON hg.hostid = h.hostid
            WHERE h.status = 0 AND h.flags IN (0, 4)
            """
        

        conditions = []

        if hostname:
            conditions.append(f"h.host = '{hostname}'")
        if host_group:
            conditions.append(f"g.name = '{host_group}'")
        if conditions:
            return base_query + " AND " + " AND ".join(conditions)
        return base_query

class ZabbixDB:
    """A class to handle Zabbix database connections and queries for host status."""
    
    VALID_STATISTICS = {'min', 'max', 'mean', 'median', 'stdev', 'sum', 'count', 'range', 'mad', 'last', 'avg'}
    
    TABLE_MAPPING = {
        0: {'history': 'history', 'trends': 'trends'},
        1: {'history': 'history_str', 'trends': None},
        2: {'history': 'history_log', 'trends': None},
        3: {'history': 'history_uint', 'trends': 'trends_uint'},
        4: {'history': 'history_text', 'trends': None}
    }
    
    def __init__(
        self,
        db_type: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        connection_timeout: int = 10,
        max_retries: int = 3
    ) -> None:
        if db_type.lower() not in ['mysql', 'postgresql']:
            raise ValueError("db_type must be 'mysql' or 'postgresql'")
            
        self.db_type = db_type.lower()
        self.db_config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password,
            'connection_timeout': connection_timeout
        }
        self.max_retries = max_retries
        self.connection = None
        self._connect()  # Initialize connection on instantiation

    def _connect(self) -> None:
        """Establish a database connection."""
        if self.connection:
            logger.debug("Connection already exists, skipping connect")
            return

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempting to connect to {self.db_type} database (attempt {attempt + 1})")
                if self.db_type == 'mysql':
                    self.connection = mysql.connector.connect(
                        **self.db_config,
                        charset='utf8mb4',
                        use_pure=True
                    )
                else:
                    self.connection = psycopg2.connect(**self.db_config)
                logger.info("Database connection established successfully")
                return
            except (MySQLError, PostgresError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise ZabbixDBError(f"Failed to connect after {self.max_retries} attempts: {str(e)}")
                time.sleep(1)

    def _ensure_connection(self) -> None:
        """Ensure the connection is active, reconnect if necessary."""
        try:
            if self.db_type == 'mysql':
                if not self.connection or not self.connection.is_connected():
                    logger.debug("MySQL connection is closed, reconnecting")
                    self.connection = None
                    self._connect()
            else:  # PostgreSQL
                if not self.connection or self.connection.closed:
                    logger.debug("PostgreSQL connection is closed, reconnecting")
                    self.connection = None
                    self._connect()
        except (MySQLError, PostgresError) as e:
            logger.error(f"Failed to ensure connection: {str(e)}")
            raise ZabbixDBError(f"Connection error: {str(e)}")

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            try:
                if self.db_type == 'mysql' and self.connection.is_connected():
                    logger.debug("Closing MySQL connection")
                    self.connection.close()
                elif self.db_type == 'postgresql' and not self.connection.closed:
                    logger.debug("Closing PostgreSQL connection")
                    self.connection.close()
                logger.info("Database connection closed")
            except (MySQLError, PostgresError) as e:
                logger.error(f"Error closing connection: {str(e)}")
            finally:
                self.connection = None

    def __del__(self):
        """Ensure connection is closed when instance is destroyed."""
        self.close()

    def compute_statistic(self, data: List[Dict[str, Any]], operation: str) -> Union[List[Dict[str, Any]], float, int]:
        """Compute statistical measure on list of dicts."""
        if not data:
            return []
            
        if operation not in self.VALID_STATISTICS:
            raise ZabbixDBError(f"Unsupported operation: {operation}")
            
        values = []
        for item in data:
            try:
                values.append(round(float(item['value']), 2))
            except (TypeError, ValueError):
                values.append(item['value'])  # Keep original value if conversion fails
        

        if not values:
            return []
        
        if operation == 'min':
            min_value = min(values)
            return [item for item in data if round(float(item['value']), 2) == min_value]
        elif operation == 'max':
            max_value = max(values)
            return [item for item in data if round(float(item['value']), 2) == max_value]
        elif operation == 'last':
            return [max(data, key=lambda x: x['clock'])]
        elif operation in ('mean', 'avg'):
            return statistics.fmean(values)
        elif operation == 'median':
            return statistics.median(values)
        elif operation == 'stdev' and len(values) > 1:
            return statistics.stdev(values)
        elif operation == 'sum':
            return sum(values)
        elif operation == 'count':
            return len(values)
        elif operation == 'range':
            return max(values) - min(values)
        elif operation == 'mad':
            med = statistics.median(values)
            return statistics.median([abs(x - med) for x in values])
        return []

    def time_difference(self, time_from: int, time_to: int) -> int:
        """Calculate difference in days between two Unix timestamps."""
        try:
            dt_from = datetime.fromtimestamp(time_from, tz=timezone.utc)
            dt_to = datetime.fromtimestamp(time_to, tz=timezone.utc)
            return (dt_to - dt_from).days
        except ValueError as e:
            raise ZabbixDBError(f"Invalid timestamp: {str(e)}")

    def convert_day(self, duration: str) -> float:
        """Convert duration string to days."""
        try:
            matches = re.findall(r'(\d+)([dhm])', duration.lower())
            total_days = 0.0
            for value, unit in matches:
                value = int(value)
                if unit == 'd':
                    total_days += value
                elif unit == 'h':
                    total_days += value / 24
                elif unit == 'm':
                    total_days += value / (24 * 60)
            return round(total_days, 2)
        except (ValueError, TypeError) as e:
            raise ZabbixDBError(f"Invalid duration format: {str(e)}")

    def get_monitoring_status(self, hostname: str) -> int:
        """Get monitoring status for a host."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(QueryBuilder.build_host_status_query(), (hostname,))
                result = cursor.fetchone()
                return 1 if result and result['status'] == 1 else 0
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_monitoring_status: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_host_by_group(self, hostname:str = None,host_group: str = None) -> List[Dict[str, Any]]:
        """Get hosts by group name."""
        self._ensure_connection()
        

        query = QueryBuilder.build_host_by_group_query(host_group=host_group,hostname=hostname)
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(query)
                return cursor.fetchall() or []
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_host_by_group: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_item_detail(self, itemid: int = None,item_name: str = None, hostname: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Fetch item details."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_item_detail_query(itemid=itemid, item_name=item_name, hostname=hostname)
                             
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return None
                    
                def map_item(item):
                    value_type = item['value_type']
                    if value_type not in self.TABLE_MAPPING:
                        raise ZabbixDBError(f"Invalid value_type {value_type}")
                    return {
                        'hostname': item['host'],
                        'itemid': item['itemid'],
                        'master_itemid': item.get('master_itemid', None),
                        'hostid': item['hostid'],
                        'name': item['name'],
                        'delay': item['delay'],
                        'history': item['history'],
                        'trends': item['trends'],
                        'value_type': value_type,
                        'status': item['status'],
                        'units': item['units'],
                        'history_table_name': self.TABLE_MAPPING[value_type]['history'],
                        'trends_table_name': self.TABLE_MAPPING[value_type]['trends']
                    }
                return map_item(results[0]) if hostname or itemid else [map_item(item) for item in results]
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_item_detail: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_trend_data(self, itemid: str, time_from: int, time_to: int, trend_table_name: str, statistical_measure: str = None):
        """Fetch trend data."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_trend_query(trend_table_name, statistical_measure or 'avg')
                cursor.execute(query, (itemid, time_from, time_to))
                result = cursor.fetchall() or []
                
                if statistical_measure and statistical_measure != 'all':
                    return self.compute_statistic(result, statistical_measure)
                return result
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_trend_data: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_history_data(self, itemid: str, time_from: int, time_to: int, history_table_name: str, statistical_measure: str = None):
        """Fetch history data."""
        self._ensure_connection()
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_history_query(history_table_name)
                cursor.execute(query, (itemid, time_from, time_to))
                result = cursor.fetchall() or []
                
                if statistical_measure:
                    return self.compute_statistic(result, statistical_measure)
                return result
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_history_data: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_function_name(self, time_from: int, time_to: int, history_days: float, trends_days: float) -> str:
        """Determine appropriate data fetch function."""
        current_time = float(datetime.now().timestamp())
        seconds_in_day = 86400.0
        history_threshold = int(current_time - (history_days * seconds_in_day))
        trends_threshold = int(current_time - (trends_days * seconds_in_day))

        if trends_days == 0:
            return "get_history"
        if time_to < trends_threshold:
            return "No data - too old"

        elif time_from >= history_threshold:
            return "get_history"
        elif time_to <= history_threshold and time_from >= trends_threshold:
            return "get_trend"
        elif time_from < history_threshold and time_to >= history_threshold:
            return "get_trend"
        return "Invalid range"

    def get_metric_data(self, hostname: str, metric_name: str, time_from: int, time_to: int, statistical_measure: str = None):
        """Fetch metric data."""
        if time_from > time_to:
            return self._error_response(
                f"Invalid time range: time_from={time_from} > time_to={time_to}",
                hostname, metric_name, "unknown", statistical_measure
            )

        try:
            monitoring_status = self.get_monitoring_status(hostname)
            item_details = self.get_item_detail(item_name=metric_name, hostname=hostname)

            if not item_details:
                return self._error_response(
                    f"Item '{metric_name}' not found for host '{hostname}'",
                    hostname, metric_name, "unknown", statistical_measure
                )

            if monitoring_status == 1:
                return self._error_response(
                    f"Host '{hostname}' is disabled",
                    hostname, metric_name, "unknown"
                )

            if item_details['status'] != 0:
                return self._error_response(
                    f"Item '{metric_name}' is disabled",
                    hostname, metric_name, item_details['units']
                )

            history_table = item_details['history_table_name']
            trends_table = item_details['trends_table_name']

            if not history_table:
                return self._error_response(
                    f"No valid history table found for item '{metric_name}' with value_type {item_details['value_type']}",
                    hostname, metric_name, item_details['units'], statistical_measure
                )

            if history_table in ['history_str', 'history_log', 'history_text']:
                function_name = "get_history"
                statistical_measure = statistical_measure if statistical_measure == 'last' else None
                
            else:
                function_name = self.get_function_name(
                    time_from,
                    time_to,
                    self.convert_day(item_details['history']),
                    self.convert_day(item_details['trends'])
                )

            if function_name in ["No data - too old", "Invalid range"]:
                return self._error_response(
                    f"Cannot fetch data: {function_name}",
                    hostname, metric_name, item_details['units'], statistical_measure
                )

            if statistical_measure and statistical_measure not in self.VALID_STATISTICS:
                return self._error_response(
                    f"Invalid statistical measure: {statistical_measure}",
                    hostname, metric_name, item_details['units'], statistical_measure
                )

            fetch_func = {
                "get_history": lambda: self.get_history_data(
                    item_details['itemid'],
                    time_from,
                    time_to,
                    history_table,
                    statistical_measure
                ),
                "get_trend": lambda: self.get_trend_data(
                    item_details['itemid'],
                    time_from,
                    time_to,
                    trends_table,
                    statistical_measure
                )
            }

            data = fetch_func[function_name]()
            return self._success_response(
                data=data,
                hostname=hostname,
                metric_name=metric_name,
                unit=item_details['units'],
                statistical_measure=statistical_measure
            )

        except ZabbixDBError as e:
            logger.error(f"Error in get_metric_data: {str(e)}")
            return self._error_response(
                str(e), hostname, metric_name,
                item_details['units'] if item_details else "unknown",
                statistical_measure
            )

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        """Fetch all alerts."""
        self._ensure_connection()
        
        query = """
        SELECT DISTINCT
            h.name AS host,
            t.description AS trigger_name,
            e.name AS event_name,
            CASE e.severity
                WHEN 0 THEN 'Not classified'
                WHEN 1 THEN 'Information'
                WHEN 2 THEN 'Warning'
                WHEN 3 THEN 'Average'
                WHEN 4 THEN 'High'
                WHEN 5 THEN 'Disaster'
            END AS severity,
            e.eventid,
            e.acknowledged,
            e.clock AS start_time,
            COALESCE(er.clock, UNIX_TIMESTAMP()) AS end_time,
            COALESCE(er.clock, UNIX_TIMESTAMP()) - e.clock AS duration,
            er.eventid AS recovery_eventid
        FROM hosts h
        JOIN items i ON i.hostid = h.hostid AND i.status = 0
        JOIN functions f ON f.itemid = i.itemid
        JOIN triggers t ON t.triggerid = f.triggerid AND t.status = 0
        JOIN events e ON e.objectid = t.triggerid AND e.object = 0 AND e.value = 1
        LEFT JOIN event_recovery erc ON erc.eventid = e.eventid
        LEFT JOIN events er ON er.eventid = erc.r_eventid
        WHERE 
            h.status = 0
            AND h.flags IN (0, 4)
        """
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(query)
                return cursor.fetchall() or []
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_all_alerts: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

    def get_alerts(self, time_from: int = None, time_to: int = None, hostname: str = None, limit: int = None, host_group: str = None):
        """Filter alerts based on various parameters."""
        try:
            alerts = pd.DataFrame(self.get_all_alerts())
            if alerts.empty:
                return []

            alerts = alerts.sort_values(by='start_time', ascending=False)

            if time_from is not None:
                alerts = alerts[alerts['start_time'] >= time_from]
            if time_to is not None:
                alerts = alerts[alerts['start_time'] <= time_to]
            if hostname is not None:
                alerts = alerts[alerts['host'] == hostname]
            if host_group is not None:
                hosts = self.get_host_by_group(host_group)
                host_names = [host['host_name'] for host in hosts]
                alerts = alerts[alerts['host'].isin(host_names)]

            if limit is not None:
                alerts = alerts.head(limit)

            return alerts.to_dict(orient='records')

        except Exception as e:
            logger.error(f"Error in get_alerts: {str(e)}")
            raise ZabbixDBError(f"Failed to process alerts: {str(e)}")

    def get_common_issues(self, time_from: int = None, time_to: int = None, hostname: str = None, limit: int = None, host_group: str = None):
        """Get common issues aggregated by event name."""
        try:
            alerts = self.get_alerts(time_from, time_to, hostname, host_group)
            if not alerts:
                return self._success_response(
                    data=[],
                    hostname=hostname,
                    time_from=time_from,
                    time_to=time_to,
                    limit=limit,
                    host_group=host_group
                )

            common_issues = pd.DataFrame(alerts).groupby('event_name').agg(
                total_count=('eventid', 'count'),
                acknowledged_count=('acknowledged', lambda x: (x == 1).sum()),
                unacknowledged_count=('acknowledged', lambda x: (x == 0).sum())
            ).reset_index().sort_values(by='total_count', ascending=False)

            if limit is not None:
                common_issues = common_issues.head(limit)

            return self._success_response(
                data=common_issues.to_dict(orient='records'),
                hostname=hostname,
                time_from=time_from,
                time_to=time_to,
                limit=limit,
                host_group=host_group
            )

        except Exception as e:
            logger.error(f"Error in get_common_issues: {str(e)}")
            return self._error_response(
                f"Failed to compute common issues: {str(e)}",
                hostname, None, None, None,
                time_from=time_from, time_to=time_to, limit=limit, host_group=host_group
            )

    def get_host_by_metric(self, metric_name: str, statistical_measure: str = 'last', time_from: int = None, time_to: int = None, limit: int = None):
        """Get hosts by metric values."""
        try:
            item_details = self.get_item_detail(item_name=metric_name)
            if not item_details:
                return self._success_response(
                    data=[],
                    metric_name=metric_name
                )
                
            if not isinstance(item_details, list):
                item_details = [item_details]

            rows = []
            hostnames = {item['hostname'] for item in item_details}
            for hostname in hostnames:
                metric_data = self.get_metric_data(
                    hostname=hostname,
                    metric_name=metric_name,
                    time_from=time_from,
                    time_to=time_to,
                    statistical_measure=statistical_measure
                )

                if metric_data and metric_data.get("status") == "success":
                    data_points = metric_data.get("data")
                    logger.debug(f"Data points for host {hostname}: {data_points}, type: {type(data_points)}")
                    if isinstance(data_points, list) and data_points and isinstance(data_points[0], dict):
                        for point in data_points:
                            try:
                                value = float(point.get("value"))
                            except (TypeError, ValueError):
                                value = None
                            rows.append({
                                "hostname": metric_data.get("hostname"),
                                "unit": metric_data.get("unit"),
                                "clock": point.get("clock"),
                                "value": value
                            })
                    else:
                        value = None
                        if isinstance(data_points, (int, float)):
                            value = float(data_points)
                        elif isinstance(data_points, list):
                            if len(data_points) == 1:
                                try:
                                    value = float(data_points[0])
                                except (TypeError, ValueError):
                                    pass
                            elif len(data_points) == 0:
                                logger.debug(f"No data points for host {hostname} and metric {metric_name}")
                        rows.append({
                            "hostname": hostname,
                            "unit": metric_data.get("unit"),
                            "clock": None,
                            "value": value
                        })

            if not rows:
                return self._success_response(
                    data=[],
                    metric_name=metric_name
                )

            df = pd.DataFrame(rows, columns=['hostname', 'unit', 'clock', 'value'])
            df['clock'] = pd.to_numeric(df['clock'], errors='coerce').astype('Int64')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.sort_values(by='value', ascending=False, na_position='last')

            if limit is not None:
                df = df.head(limit)

            return self._success_response(
                data=df.to_dict(orient='records'),
                metric_name=metric_name
            )

        except Exception as e:
            logger.error(f"Error in get_host_by_metric: {str(e)}")
            return self._error_response(
                f"Failed to get hosts by metric: {str(e)}",
                hostname=None,
                metric_name=metric_name,
                unit="unknown",
                statistical_measure=statistical_measure
            )

    def get_tag_for_host(self, hostname: str= None,tag_name: str = None,tag_value: str = None) -> List[Dict[str, str]]:
        """Get tags for a specific host."""
        self._ensure_connection()
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_host_by_tag_query(hostname, tag_name, tag_value)
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return {"status": "error", "message": f"No tags found for host '{hostname}'"}
                    
                return {
                    "status": "success",
                    "data": results,
                    "hostname": hostname
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_tag_for_host: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "hostname": hostname}

    def get_tag_for_item(self, item_name: str = None, hostname: str = None, tag_name: str = None, tag_value: str = None) -> List[Dict[str, str]]:
        """Get tags for a specific item."""
        self._ensure_connection()
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_item_by_tag_query(item_name, hostname, tag_name, tag_value)
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return {"status": "error", "message": f"No tags found for item '{item_name}'"}
                    
                return {
                    "status": "success",
                    "data": results,
                    "item_name": item_name,
                    "hostname": hostname
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_tag_for_item: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "item_name": item_name, "hostname": hostname}

    def get_tag_for_problem(self,eventid: int = None,event_name: str = None, tag_name: str = None, tag_value: str = None) -> List[Dict[str, str]]:
        self._ensure_connection()
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_tag_for_problem_query(eventid, event_name, tag_name, tag_value)
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return {"status": "error", "message": f"No tags found for item '{event_name}'"}
                    
                return {
                    "status": "success",
                    "data": results,
                    "event_name": event_name,
                    "eventid": eventid
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_tag_for_item: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "event_name": event_name, "eventid": eventid}

    def get_macro_for_host(self, hostname: str = None, macro_name: str = None) -> List[Dict[str, str]]:
        """Get macros for a specific host."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_macro_host_query(hostname, macro_name)
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return {"status": "error", "message": f"No macros found for host '{hostname}'"}
                    
                return {
                    "status": "success",
                    "data": results,
                    "hostname": hostname
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_macro_for_host: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "hostname": hostname}

    def get_acknowledgement(self, eventid: list[int]) -> Dict[str, Any]:
        """Get acknowledgement details for a specific event."""
        self._ensure_connection()
        
        placeholders = ','.join(['%s'] * len(eventid))

        query = f"""
        SELECT ack.eventid, ack.clock, ack.message, u.name
        FROM acknowledges ack
        JOIN users u ON u.userid = ack.userid
        WHERE ack.eventid IN ({placeholders})
        """
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(query,eventid)
                result = cursor.fetchall()
                
                if not result:
                    return {"status": "No Data","data": result,"eventid": eventid}
                    
                return {
                    "status": "success",
                    "data": result,
                    "eventid": eventid
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_acknowledgement: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "eventid": eventid}

    def get_host_ip(self, hostname: str) -> Dict[str, Any]:
        """Get IP address for a specific host."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = """
                SELECT h.hostid, h.host, i.ip,i.dns,i.available
                FROM hosts h
                JOIN interface i ON i.hostid = h.hostid
                WHERE h.host = %s
                """
                cursor.execute(query, (hostname,))
                result = cursor.fetchone()
                
                if not result:
                    return self._error_response(
                        f"Host '{hostname}' not found",
                        hostname=hostname
                    )
                    
                return self._success_response(
                    data={
                        "hostid": result['hostid'],
                        "hostname": result['host'],
                        "ip": result['ip'],
                        "dns": result['dns'],
                        "available": result['available']
                    },
                    hostname=hostname
                )
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_host_ip: {str(e)}")
            return {"status": "error", "message": f"Query failed: {str(e)}", "hostname": hostname}

    def get_host_drives(self, hostname: str) -> Dict[str, Any]:
        """Fetch item details."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                query = QueryBuilder.build_item_detail_query(hostname=hostname)
                             
                cursor.execute(query)
                results = cursor.fetchall()
                
                if not results:
                    return None
                return {
                    "status": "success",
                    "data": results,
                    "hostname": hostname
                }
        
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_host_drives: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")


    def _error_response(self, message: str, hostname: str = None, metric_name: str = None, unit: str = None, statistical_measure: str = None, **kwargs):
        """Create error response dictionary."""
        response = {
            "status": "error",
            "message": message,
            "hostname": hostname,
            "metric_name": metric_name,
            "unit": unit,
            "data": [],
            "statistical_measure": statistical_measure
        }
        response.update({k: v for k, v in kwargs.items() if v is not None})
        return response

    def _success_response(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Create success response dictionary."""
        base_response = {
            "status": "success",
            "data": data,
            **kwargs
        }
        return {k: v for k, v in base_response.items() if v is not None}

    def get_host_status(self, hostname: str) -> Union[Dict[str, Any], str]:
        """Fetch host status."""
        self._ensure_connection()
        
        try:
            with self.connection.cursor(dictionary=True) as cursor:
                cursor.execute(QueryBuilder.build_host_status_query(), (hostname,))
                result = cursor.fetchone()
                
                if not result:
                    return f"Host '{hostname}' not found"
                    
                if result['status'] == 1:
                    return f"Host '{hostname}' is disabled"
                    
                return {
                    'hostid': result['hostid'],
                    'hostname': result['host'],
                    'status': result['status']
                }
        except (MySQLError, PostgresError) as e:
            logger.error(f"Query failed in get_host_status: {str(e)}")
            raise ZabbixDBError(f"Query failed: {str(e)}")

if __name__ == "__main__":
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        db_config = config['database_config']

    try:
        zbx = ZabbixDB(**db_config)
        result = zbx.get_host_by_metric(
            metric_name='CPU utilization',
            time_from=1749032410,
            time_to=1749118810,
            statistical_measure='avg'
        )
        print(result)
        zbx.close()
    except ZabbixDBError as e:
        print({"status": "error", "message": str(e)})