from flask_mysqldb import MySQL, MySQLdb
from app import app

class cam_db:
  def __init__(self):
    self.mysql = MySQL(app)

  def getall_cam(self):
    cur = self.mysql.connection.cursor()
    cur.execute("SELECT id, nama FROM cctv")
    data = list(cur.fetchall())
    print("Isi data", data)
    cur.close()

    return data
  
  def get_name_byid(self, id):
    cur = self.mysql.connection.cursor()
    cur.execute("SELECT nama FROM cctv WHERE id=%s", (id,))
    data = str(cur.fetchone()[0])
    cur.close()

    return data

  def get_url_byid(self, id):
    cur = self.mysql.connection.cursor()
    cur.execute("SELECT url FROM cctv WHERE id=%s", (id,))
    data = str(cur.fetchone()[0])
    cur.close()

    return data
