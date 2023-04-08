# @Time: 2022/11/25 9:45
import os

import app


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sorry, it is impossible to guess'

