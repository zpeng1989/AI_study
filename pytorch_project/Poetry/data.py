import sys
import os
import json
import re
import numpy as np


def _parseRawData(author = None, constrain = None, str = './chinese-poetry/json/simplified', category = 'poet.tang'):

