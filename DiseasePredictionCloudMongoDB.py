from pymongo import MongoClient

Client= MongoClient("mongobd+srv//username: password@cluster.mongodb.net/")
db= client["medical_ai"]
feedback_collection= db["user_feedback"]

@app.route(' /feedback', methods= ['POST'])
def feedback():
    data= request.json

feedback_collection.insert_one(data)
    return jsonify({"message": "Feedback received!"})
                                                                                                                 
