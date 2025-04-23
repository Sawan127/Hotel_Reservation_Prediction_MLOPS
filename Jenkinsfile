pipeline{
    agent any
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps {
                script{
                    echo 'Cloning Github repo to Jenkins'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Sawan127/Hotel_Reservation_Prediction_MLOPS.git']])
                    
                }

            }
        }
    }
}