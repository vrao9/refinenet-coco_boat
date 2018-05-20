function test_load_files()

epoch_no = 0;
path = 'D:\TUHH\Arbeit\result\model_20180511162038\model_cache';
files  = epoch_names(45);

for f = files
    epoch_no = epoch_no + 5;
    mat_file = strcat('net-config-','epoch-',int2str(epoch_no),'.mat');
    net_conf = strcat(path,'\',f,'\',mat_file); 
    load(net_conf{1});
    disp(epoch_no)
end
end


function s = epoch_names(epochs_no)
%s = {epoch_no};
for c = 1:epochs_no
    s{c} = strcat('epoch_',int2str(5*c));
end
end