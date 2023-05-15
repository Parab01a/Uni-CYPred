# @Time: 2023/5/8 19:17
import pymysql


class MysqlUtil:
    def __init__(self):
        self.db = pymysql.connect(user='root',
                                  password='florentinoariza516',
                                  host='localhost',
                                  port=3306,
                                  database='db_p450_prediction')
        self.cursor = self.db.cursor()
        # print(self.connect) 检查MySQL是否连接成功

    def insert(self, sql):
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print("error: ", e)
            self.db.rollback()
        finally:
            self.db.close()

    def select(self, sql, uuid_str=None):
        try:
            self.cursor.execute(sql, uuid_str)
            return self.cursor.fetchone()
        except Exception as e:
            print("error: ", e)
            self.db.rollback()
        finally:
            self.db.close()

    def insert_file(self, sql, csv_file=None):
        try:
            self.cursor.execute(sql, csv_file)
            self.db.commit()
        except Exception as e:
            print("error: ", e)
            self.db.rollback()
        finally:
            self.db.close()

    def close(self):
        self.cursor.close()
        self.db.close()

# if __name__ == '__main__':
#     Mysql = MysqlUtil()
#     sql = 'select * from prediction_results'
#     print(Mysql.db_action(sql=sql, flag=1))
#     sql = "insert into prediction_results values(null, '{}', '{}', NOW())".format('1', '1')
#     print(Mysql.db_action(sql=sql, flag=0))
