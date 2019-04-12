    import json
    import requests
    from flask import Flask
    from flask import request
    from flask import Response
    import re
    
    
    
    token ='895114946:AAFY3GxIK6iNxECUN2VQgFfJK3-KdQ9W55M'
    cmc_token = 'be78de6a-49fa-44ed-8e58-f68aff2923ae'
    
    app = Flask(__name__)
    
    
    
    def write_json(data, filename='response.json'):
            with open(filename,'w') as f:
                json.dump(data,f,indent=4,ensure_ascii= False)
                
    def get_cmc_data(crypto):
        url ='https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        params = {'symbol':crypto,'convert':'USD'}
        headers = {'X-CMC_PRO_API_KEY':cmc_token}
    
        r = requests.get(url,headers=headers,params=params).json()
        
       # print(r)
       # write_json(r)
        price = r['data'][crypto]['quote']['USD']['price']
    
        return price            
      
        
    
    def parse_message(message):
        chat_id =message['message']['chat']['id']
        txt = message['message']['text']
        pattern = r'/[a-zA-Z]{2,4}'
        ticker = re.findall(pattern,txt)
        
        if ticker:
            symbol = ticker[0][1:].upper()
            
        else:
            symbol =''
            
        return chat_id,symbol    
            
    
    def send_message(chat_id,text = 'bla-bla-bla'):
        url = f'https://api.telegram.org/bot{token}/sendMessage'
        payload = {'chat_id':chat_id,'text':text}
        r = requests.post(url,json = payload)
        return r
    
    
    
    @app.route('/',methods=['POST','GET'])
    def index():
        if request.method == 'POST':
            msg = request.get_json()
            
            chat_id,symbol = parse_message(msg)
            
            if not symbol:
                send_message(chat_id,'Wrong data')
                return Response('Ok',status = 200)
            
            price = get_cmc_data(symbol)
            
            send_message(chat_id,price)
           # write_json(msg,'telegram_request.json')
            return Response('Ok',status = 200)
        else:
            return '<h1>CoinMarketCap bot</h1>'
    
        
        
    def main():
       print (get_cmc_data('BTC'))    
        
    
    if __name__ == '__main__':
       
        
        # main()
      
       app.run(port=5000, debug=True)
        
        #TO DO BOT
        #1. locally create  a basic flask application
        #2.Set up a tunnel
        #3.Set a webhook
        #4.Receive and parse a user message
        #5.Send message to user
        
        
        
        #https://api.telegram.org/bot895114946:AAFY3GxIK6iNxECUN2VQgFfJK3-KdQ9W55M/getMe
        #https://api.telegram.org/bot895114946:AAFY3GxIK6iNxECUN2VQgFfJK3-KdQ9W55M/getUpdates
        #https://api.telegram.org/bot895114946:AAFY3GxIK6iNxECUN2VQgFfJK3-KdQ9W55M/sendMessage?chat_id=769857735&text=Hello User
        #https://api.telegram.org/bot895114946:AAFY3GxIK6iNxECUN2VQgFfJK3-KdQ9W55M/setWebhook?url=https://primum.serveo.net
