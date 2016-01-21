import sqlite3

class DBController :
	db = None

	def __init__(self,dbname):
		self.open_db(dbname)
		self.initiate_table_uds()
		self.initiate_table_idmap()

	def open_db(self,dbname):
		self.db = sqlite3.connect(dbname) 

	def close_db(self):
		self.db.close()
	
	def initiate_table_uds(self):
		try:
			cursor = self.db.cursor()
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS uds(id INTEGER PRIMARY KEY, uds_string TEXT)  
			''')
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def initiate_table_idmap(self):
		try:
			cursor = self.db.cursor()
			cursor.execute('''
				CREATE TABLE IF NOT EXISTS idmap(id INTEGER PRIMARY KEY, filename TEXT)  
			''')
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def drop_table_uds(self):
		try:
			cursor = self.db.cursor()
			cursor.execute('''
				DROP TABLE uds  
			''')
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def drop_table_idmap(self):
		try:
			cursor = self.db.cursor()
			cursor.execute('''
				DROP TABLE idmap
			''')
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def start_over_tables(self):
		self.drop_table_uds
		self.drop_table_idmap
		self.initiate_table_uds()
		self.initiate_table_idmap()

	def insert_new_uds_file(self, id_var, filename_var, uds_var):
		try:
			cursor = self.db.cursor()
			cursor.execute('''
				INSERT INTO uds(id, uds_string)
	            VALUES(?,?)''', (id_var,uds_var)) 
			cursor.execute('''
				INSERT INTO idmap(id, filename)
	            VALUES(?,?)''', (id_var,filename_var)) 
			self.db.commit()
		# except IntegrityError as e:
		# 	print("already exist")
		except Exception as e:
			self.db.rollback()
			raise e

	def get_uds_file_list(self):
		cursor = self.db.cursor()
		cursor.execute('''SELECT id, filename FROM idmap''')
		all_rows = cursor.fetchall()
		return all_rows

	def get_filename_from_id(self,id_var):
		cursor = self.db.cursor()
		cursor.execute('''SELECT filename FROM idmap WHERE id=?''', (id_var,))
		filename = cursor.fetchone()
		return filename

	def get_uds_string_from_id(self,id_var):
		cursor = self.db.cursor()
		cursor.execute('''SELECT uds_string FROM uds WHERE id=?''', (id_var,))
		uds_string = cursor.fetchone()
		return uds_string

	def get_uds_string_from_filename(self,filename_var):
		cursor = self.db.cursor()
		cursor.execute('''SELECT id FROM idmap WHERE filename=?''', (filename_var,))
		id_var = int(cursor.fetchone())
		cursor.execute('''SELECT uds_string FROM uds WHERE id=?''', (id_var,))
		uds_string = cursor.fetchone()
		return uds_string

	def update_uds_string_from_id(self,id_var, uds_var):
		try:
			cursor = self.db.cursor()
			cursor.execute('''UPDATE uds SET uds_string = ? WHERE id=?''',
			(uds_var, id_var))
			self.db.commit()
		except Exception as e:
			db.rollback()
			raise e
	def update_uds_string_from_filename(self,filename_var,uds_var):
		try:
			cursor = self.db.cursor()
			cursor.execute('''SELECT id FROM idmap WHERE filename=?''', (filename_var,))
			id_var = int(cursor.fetchone())
			cursor.execute('''UPDATE uds SET uds_string = ? WHERE id=?''',
			(uds_var, id_var))
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def delete_uds_string_from_id(self,id_var):
		try:
			cursor = self.db.cursor()
			cursor.execute('''DELETE FROM uds WHERE id = ? ''', (id_var,))
			self.self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e

	def delete_uds_string_from_filename(self,filename_var):
		try:
			cursor = self.db.cursor()
			cursor.execute('''SELECT id FROM idmap WHERE filename=?''', (filename_var,))
			id_var = int(cursor.fetchone())
			cursor.execute('''DELETE FROM uds WHERE id = ? ''', (id_var,))
			self.db.commit()
		except Exception as e:
			self.db.rollback()
			raise e