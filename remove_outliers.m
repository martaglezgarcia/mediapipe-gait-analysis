function output_data = remove_outliers(input_data)
	% Remove outliers
	output_data = rmoutliers(input_data,"gesd",...
	    "DataVariables","angle");
end