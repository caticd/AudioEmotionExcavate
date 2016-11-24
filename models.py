from sqlalchemy import create_engine, Column, Integer, VARCHAR, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import time

Base = declarative_base()
sql_engine = create_engine('mysql://emotion:emotion@123.206.79.187/emotion?charset=utf8')
SqlSession = sessionmaker(bind=sql_engine)


class Files(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True)
    origin_name = Column(VARCHAR(100), nullable=False)
    store_name = Column(VARCHAR(100), nullable=False)
    date_time = Column(DateTime, nullable=False)
    progress = Column(Integer, nullable=False)
    result = Column(VARCHAR(20))
    possible_angry = Column(Float)
    possible_fear = Column(Float)
    possible_happy = Column(Float)
    possible_neutral = Column(Float)
    possible_sad = Column(Float)
    possible_surprise = Column(Float)

    def __init__(self, origin_name, store_name, progress):
        self.origin_name = origin_name
        self.store_name = store_name
        self.progress = progress
        self.date_time = datetime.datetime.fromtimestamp(time.time())

    def __repr__(self):
        return '<File %r,origin_name:%r,store_name:%r,progress:%r>' % (
            self.id, self.origin_name, self.store_name, self.progress)
