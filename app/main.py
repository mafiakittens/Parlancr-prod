# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: update to connect to shakespeare endpoints per updated
# method "predict"

import logging

from flask import Flask
from flask import request

import model_loader
#from style_transfer import Model

app = Flask(__name__)
modelLdr = ModelLoader()

# This function calls "toShakespeare"
@app.route('/toShakespeare', methods=['GET'])
def shakespeareText():
    """Given an input sentence, return the same sentence in Shakespeare style."""
    #while True:
    modern_text = request.args.get('modernText')
    
    if modern_text is None:
        return 'No text input detected.', 400
    
    output_text = modelLdr.toShakespeare('tmp/shakes_mod.vocab', modern_text)
    
    if output_text is None:
        return 'You have stumped the style translator!', 400
    
    return output_text, 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
