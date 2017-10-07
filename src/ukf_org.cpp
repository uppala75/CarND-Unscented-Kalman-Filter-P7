#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //set state dimension
  n_x_ = 5; 
  //set augmented dimension
  n_aug_ = 7; 
  //int n_z_ = 3;  //set measurement dimension. radar measures r, phi & r_dot
  //sigma point spreading parameter
  lambda_ = 3 - n_aug_; 
  //predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1); 
  weights_ = VectorXd(2*n_aug_+1);
  //current NIS for radar
  NIS_radar_ = 0.0; 
  //current NIS for laser
  NIS_laser_ = 0.0; 

  is_initialized_ = false;
  time_us_ = 0.0; 

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  //***************
  // INITIALIZATION
  //***************


  if (!is_initialized_){
	//Initialize x_, P_, previous_time, anything else needed
	x_ << 1.0, 1.0, 0.0, 0.0, 0.0;

	//Initialize covariance matrix P_
	P_ << 1, 0, 0, 0, 0,
			0, 1, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;

	if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
	//if (meas_package.sensor_type_ == MeasurementPackage::LASER){
	//Initialize
		cout << "Receiving LASER Data: " << endl;
		x_(0) = meas_package.raw_measurements_(0);
		x_(1) = meas_package.raw_measurements_(1);
		cout << "End Receiving LASER Data: " << endl;
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
	//else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
	//Initialize
		cout << "Receiving RADAR Data: " << endl;
		float ro = meas_package.raw_measurements_(0);
		float phi = meas_package.raw_measurements_(1);
		float ro_dot = meas_package.raw_measurements_(2);
		x_(0) = ro*cos(phi);
		x_(1) = ro*sin(phi);
		cout << "End Receiving RADAR Data: " << endl;
	}
	//Initialize remaining parameters

	// Initialize weights
  	double weight_0 = lambda_/(lambda_+n_aug_);
  	weights_(0) = weight_0;
  	for (int i=1; i<2*n_aug_+1; i++)
  	{
  		double weight = 0.5/(n_aug_+lambda_);
  		weights_(i) = weight;
  	}

	time_us_ = meas_package.timestamp_;
	//Xsig_pred_.fill(0.0);
	is_initialized_ = true;

	return;
  }

  
  double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  //***************
  // Prediction
  //***************
  Prediction(dt);

  //***************
  // Update
  //***************

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
  	cout << "LIDAR Update begin " << endl;
  	UpdateLidar(meas_package);
  	cout << "LIDAR Update end " << endl;

  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
  	cout << "RADAR Update begin " << endl;
  	UpdateRadar(meas_package);
  	cout << "RADAR Update end " << endl;
  }

 }

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  
  
  /*****************************************************************************
  *  Generate Sigma Points
  ****************************************************************************/
  /*
  //Create a sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //Sq root of P
  MatrixXd A = P_.llt().matrixL();

  //set lambda for non-augmented sigma points
  lambda_ = 3 - n_x_;

  //set first column of sigma point matrix
  Xsig.col(0) = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }
  */

  // ****************************
  // Create Augmented Sigma points
  // Lesson 7.18: Augmentation assignment 2
  // ****************************
  //Create augmented mean state
  cout << "Prediction Start.." << endl;  
  VectorXd x_aug=VectorXd(n_aug_); //Augmented mean state vector
  x_aug.fill(0.0);
  MatrixXd P_aug=MatrixXd(n_aug_,n_aug_); //Augmented state covariance matrix
  P_aug.fill(0.0);
  MatrixXd Xsig_aug=MatrixXd(n_aug_, 2*n_aug_+1); //Augmented Sigma point matrix
  Xsig_aug.fill(0.0);

  std::cout << "n_aug_ = " << std::endl << n_aug_ << std::endl;

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  //Create augmented covariance matrix
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //Create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //Create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i=0; i < n_aug_; i++)
  {
  	Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
  	Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  //print result
  std::cout << std::endl << "AugmentedSigmaPoints" << std::endl;
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // ****************************
  // Predict Sigma points
  // Lesson 7.21: Sigma point prediction assignment 2
  // ****************************
  for (int i = 0; i<2*n_aug_+1; i++)
  {
  	double p_x = Xsig_aug(0,i);
  	double p_y = Xsig_aug(1,i);
  	double v = Xsig_aug(2,i);
  	double yaw = Xsig_aug(3,i);
  	double yawd = Xsig_aug(4,i);
  	double nu_a = Xsig_aug(5,i);
  	double nu_yawdd = Xsig_aug(6,i);

  	//predicted state values
  	double px_p, py_p;

  	//avoid div by 0
  	if (fabs(yawd) > 0.001){
  		px_p = p_x + v/yawd * (sin (yaw + yawd*delta_t) - sin(yaw));
  		py_p = p_y + v/yawd * (cos (yaw) - cos(yaw+yawd*delta_t));
  	}else{
  		px_p = p_x + v*delta_t*cos(yaw);
  		px_p = p_y + v*delta_t*sin(yaw);
  	}
  	
  	double v_p = v;
  	double yaw_p = yaw + yawd*delta_t;
  	double yawd_p = yawd;

  	//add noise
  	px_p = px_p + 0.5*nu_a*delta_t*delta_t*cos(yaw);
  	py_p = py_p + 0.5*nu_a*delta_t*delta_t*sin(yaw);
  	v_p = v_p + nu_a*delta_t;

  	yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
  	yawd_p = yawd_p + nu_yawdd*delta_t;

  	//write predicted sigma point into right column
  	Xsig_pred_(0,i) = px_p;
  	Xsig_pred_(1,i) = py_p;
  	Xsig_pred_(2,i) = v_p;
  	Xsig_pred_(3,i) = yaw_p;
  	Xsig_pred_(4,i) = yawd_p;
  }

  // ****************************
  // Predict mean & covariance
  // Lesson 7.24: assignment 2
  // ****************************
  
  // set weights
  //double weight_0 = lambda_/(lambda_+n_aug_);
  //weights_(0) = weight_0;
  //for (int i=1; i<2*n_aug_+1; i++)
  //{
  //	double weight = 0.5/(n_aug_+lambda_);
  //	weights_(i) = weight;
  //}

  //predicted state mean
  x_.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++)
  {
  	x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {  
  	//state difference
  	VectorXd x_diff = Xsig_pred_.col(i) - x_;
  	//angle normalization
  	while (x_diff(3)>M_PI) x_diff(3) -= 2.*M_PI;
  	while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

  	P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

    cout << "P_ prediction complete" << P_ << endl;
    cout << "x_ prediction complete" << x_ << endl;
 }
/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
	//Lesson 5.10-13
	//H matrix will be different. UKF will have 5 components instead of 4 - add additional zero
  int n_z = 2; // measurement dimension for x & y positions

  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);
  
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1); //sigma point matrix in measurement space
  Zsig.fill(0.0);

  //Sigma points transformed into measurement space
  for (int i = 0; i<2*n_aug_+1; i++)
  {
  	double p_x = Xsig_pred_(0,i);
  	double p_y = Xsig_pred_(1,i);

  	//measurement model
  	Zsig(0,i) = p_x;
  	Zsig(1,i) = p_y;
  }

  //mean measurement prediction
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
  	z_pred=z_pred+weights_(i)*Zsig.col(i);
  }

  //measurement co-variance matrix
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
  	//residual
  	VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI; 	

  	S = S + weights_(i)*z_diff*z_diff.transpose();
  }
  
  //Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_laspx_*std_laspx_, 0,
  		0, std_laspy_*std_laspy_;
  S = S+R;

  //Cross correlation matrix calculation
  MatrixXd Tc=MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++)
  {
  	//residual
  	VectorXd z_diff=Zsig.col(i)-z_pred;

  	//angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  	//state difference
  	VectorXd x_diff=Xsig_pred_.col(i) - x_;

  	//angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

  	Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
  }

  //Kalman filter gain
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //NIS calculation
  NIS_laser_ = z_diff.transpose()*S.inverse()*z_diff;

  //Update state mean & covariance matrix
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();

  cout<<"P after laser update:"<< endl;
  cout<<P_<< endl;

  cout<<"x after laser update:"<< endl;
  cout<<x_<< endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3; // measurement dimension for ro, phi & ro_dot

  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);

  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1); //sigma point matrix in measurement space
  Zsig.fill(0.0);

  //************************************ 
  //   Predict radar sigma points
  //************************************ 
  // Lesson 7.27: predict radar measurment assignment 2
  for (int i = 0; i<2*n_aug_+1; i++)
  {
  	double p_x = Xsig_pred_(0,i);
  	double p_y = Xsig_pred_(1,i);
  	double v = Xsig_pred_(2,i);
  	double yaw = Xsig_pred_(3,i);

  	double v1 = cos(yaw)*v;
  	double v2 = sin(yaw)*v;

  	//measurement model
  	Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y); //ro
  	Zsig(1,i) = atan2(p_y,p_x); //phi
  	Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y); //ro_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
  	z_pred = z_pred+weights_(i) * Zsig.col(i);
  }
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i<2*n_aug_+1; i++)
  {
  	//residual
  	VectorXd z_diff = Zsig.col(i) - z_pred;

  	//angle normalization
  	while (z_diff(1)>M_PI) z_diff(1) -= 2.*M_PI;
  	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  	S = S+weights_(i)*z_diff*z_diff.transpose();
  }

  //add measurement covariance matrix
  MatrixXd R=MatrixXd(n_z,n_z);
  R << std_radr_*std_radr_, 0, 0,
  		0, std_radphi_*std_radphi_, 0,
  		0, 0, std_radrd_*std_radrd_;
  S=S+R;

  //************************************ 
  //   Update radar 
  //************************************ 
  // Lesson 7.30: UKF update assignment 2

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++){

  	//residual
  	VectorXd z_diff = Zsig.col(i) - z_pred;
  	//angle normalization
  	while (z_diff(1)>M_PI) z_diff(1)-= 2.*M_PI;
  	while (z_diff(1)<-M_PI) z_diff(1)+= 2.*M_PI;

  	//state difference
  	VectorXd x_diff = Xsig_pred_.col(i) - x_;
  	//angle normalization
  	while (x_diff(3)>M_PI) x_diff(3) -= 2.*M_PI;
  	while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

  	Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain k
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)>M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

  //NIS calculation
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //update state mean & covariance matrix
  x_=x_ + K*z_diff;
  P_=P_ - K*S*K.transpose();

  cout<<"P after radar update:"<< endl;
  cout<<P_<< endl;

  cout<<"x after radar update:"<< endl;
  cout<<x_<< endl;
}
