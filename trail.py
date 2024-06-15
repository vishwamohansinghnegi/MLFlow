import mlflow
warnings.filterwarnings(...)

def calculate_sum(x , y):
    return x+y

if __name__=='__main__':
    # starting the server of mlflow
    with mlflow.start_run():
        x , y = 10 ,20
        z = calculate_sum(x,y)
        # tracking the experiment with mlflow
        mlflow.log_param('x' , x)
        mlflow.log_param('y' , y)
        mlflow.log_metric('z' , z)

        # all tracking of this file will go into 0