import os
import numpy as np
import torch.nn as nn
import torch
import warnings
import random
from collections import defaultdict

def calculate_mse(output, target, and_mae = True, return_mean = True):
    # output shape: [N,3,...]
    output = torch.clamp(output, min = 1e-10)
    mean, std_dev_no_reg, shift = output[:, 0], output[:, 1], output[:, 2]
    std_dev_regularized = std_dev_no_reg + torch.sqrt(mean) #+ 0.35 * mean

    #Obtener shape, scale y shift
    shape = mean**2 / std_dev_regularized**2
    scale = std_dev_regularized**2 / mean
    cdf_gamma = torch.igamma(shape,shift/scale) #CDF value at x = shift for gamma distrib with shape, scale parameters.a

    expected_value = mean*(1 - cdf_gamma)*(1 - torch.igamma(shape + 1,shift/scale)) - shift*(1 - cdf_gamma)**2
    mse = ((expected_value - target) ** 2)
    if and_mae:
        if return_mean:
            return mse.mean(), mse.sqrt().mean()
        else:
            return mse, mse.sqrt().mean()
    else:
        if return_mean:
            return mse.mean()
        else:
            return mse

def calculate_brier_score(output, target, threshold = 0.1, return_mean = True):
    output = torch.clamp(output, min = 1e-10)
    mean, std_dev_no_reg, shift = output[:, 0], output[:, 1], output[:, 2]
    std_dev_regularized = std_dev_no_reg + torch.sqrt(mean) #+ 0.35 * mean
    #Obtener shape, scale y shift
    shape = mean**2 / std_dev_regularized**2
    scale = std_dev_regularized**2 / mean

    target_binarized = target <= threshold
    cdf_gamma_at_thr = torch.igamma(shape, (shift + threshold)/scale)
    brier_score = (cdf_gamma_at_thr - target_binarized.float())**2 #https://www.mendeley.com/reference-manager/reader/6fe48596-cf67-3f17-b830-0fdbfba521f7/46dbbe3b-a6a3-c54b-5a62-3629c10b86e0/
    if return_mean:
        return brier_score.mean()
    else:
        return brier_score

#Implementation of CDF Gamma Backward method. It is necessary for CRPS loss function.
#Adapted from https://github.com/aesara-devs/aesara/blob/main/aesara/scalar/math.py#L678
class CDFGamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, concentration, rate_input):
        #rate_input = rate * input = input/scale
        ctx.save_for_backward(concentration, rate_input)
        result = torch.igamma(concentration,rate_input)
        #result = torch.where(rate_input>0,torch.igamma(concentration,rate_input),0.)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        concentration, rate_input = ctx.saved_tensors
        grad_rate_input = grad_output * torch.exp(-rate_input-torch.lgamma(concentration) + (concentration-1) * torch.log(rate_input))
        grad_concentration = grad_output * CDFGamma.grad_concentration(concentration,rate_input) 

        return grad_concentration, grad_rate_input

    @staticmethod
    def grad_concentration(k, x):
        result = torch.zeros_like(k)
        #If x ==0 -> result = 0
        sqrt_exp = -756 - x**2 + 60 * x
        complementary_integ_mask =  ((k < 0.8) & (x > 15)) | ((k < 12) & (x > 30)) | ((sqrt_exp > 0) & (k < torch.sqrt(sqrt_exp)))

        result[complementary_integ_mask] = -CDFGamma.gammaincc_der(k[complementary_integ_mask],x[complementary_integ_mask])

        max_iters = 5e5
        mask_others = (~complementary_integ_mask) & (x != 0)
        result[mask_others] = CDFGamma.gammainc_der(k[mask_others], x[mask_others],max_iters=max_iters)
        return result

    @staticmethod
    def gammainc_der(k,x, max_iters):
        #print(f'Using gammainc_der: {k} {x}')
        log_x = torch.log(x)
        precision = 1e-10
        k_plus_n = k.clone()
        sum_a = 0.0
        sum_b = 0.0
        log_gamma_k_plus_n_plus_1 = torch.lgamma(k + 1)
        for n in range(0, int(max_iters) + 1):
            term_a_exp = k_plus_n * log_x - log_gamma_k_plus_n_plus_1 - x
            term_a = torch.exp(term_a_exp)
            term_b = term_a * torch.special.digamma(k_plus_n + 1)

            sum_a += term_a
            sum_b += term_b

            if torch.all((term_a < precision) & (term_b < precision)):
                #print(f'n {n}')
                break

            log_gamma_k_plus_n_plus_1 += torch.log1p(k_plus_n)
            k_plus_n += 1
        #print(sum_a,sum_b)
        if n >= max_iters:
            warnings.warn(
                f"gammainc_der did not converge after {n} iterations. k={k} and x={x}",
                RuntimeWarning,
            )
        return (log_x * sum_a - sum_b)

    @staticmethod
    def gammaincc_der(k, x):
        result = torch.empty_like(k,dtype=torch.float)

        gamma_k = torch.exp(torch.lgamma(k))
        digamma_k = torch.special.digamma(k)
        log_x = torch.log(x)

        mask_asymp = (x >= k) & (x >= 8)
        k_asymp = k[mask_asymp]
        x_asymp = x[mask_asymp]

        sumatory_asympt_exp = CDFGamma.sum_asymp_expan_(k_asymp,x_asymp)
        result[mask_asymp] = (torch.special.gammaincc(k_asymp, x_asymp) * (log_x[mask_asymp] - digamma_k[mask_asymp]) + torch.exp(-x_asymp + (k_asymp - 1) * log_x[mask_asymp]) * sumatory_asympt_exp / gamma_k[mask_asymp])
       
        sumatory_grad_ser_exp = CDFGamma.sum_grad_ser_expan_(k[~mask_asymp],x[~mask_asymp])
        result[~mask_asymp] = (torch.special.gammainc(k[~mask_asymp], x[~mask_asymp]) * (digamma_k[~mask_asymp] - log_x[~mask_asymp]) + torch.exp(k[~mask_asymp] * log_x[~mask_asymp]) * sumatory_grad_ser_exp / gamma_k[~mask_asymp])
        
        return result
    
    @staticmethod
    def sum_asymp_expan_(k,x):
        # asymptotic expansion http://dlmf.nist.gov/8.11#E2
        sum = torch.zeros_like(k)
        k_minus_one_minus_n = k - 1
        fac = k_minus_one_minus_n.clone()
        dfac = torch.tensor(1.0, dtype=fac.dtype, device=fac.device)
        xpow = x.clone()
        delta = dfac / xpow

        for n in range(1, 10):
            k_minus_one_minus_n -= 1
            sum += delta
            xpow *= x
            dfac = k_minus_one_minus_n * dfac + fac
            fac *= k_minus_one_minus_n
            delta = dfac / xpow
            mask_inf = torch.isinf(delta)
            if torch.any(mask_inf):
                print('No convergió asymp_exp. Sustituyendo inf por 0')
                warnings.warn("gammaincc_der did not converge", RuntimeWarning)
            
            delta[mask_inf] = 0
        return sum

    @staticmethod
    def sum_grad_ser_expan_(k,x):
        #print(f'Using grad ser expansion:{k} {x}')
        #gradient of series expansion http://dlmf.nist.gov/8.7#E3
        sum_0 = 1/k**2 #== e^{-2*log(k)}
        n_values = torch.arange(1,1e5 + 1)  # Valores de n desde 0 hasta 100000
        n_values = n_values.float()  # Convertir a tipo float para cálculos
        # Calcular logaritmo de x, y, n_values y (n_values + k)
        log_x = torch.log(x)
        log_n_values = torch.log(n_values)
        log_n_k_values = torch.log(n_values + k.unsqueeze(-1))
        # Calcular el término (-1)^n
        neg_one_power_n = (-1) ** n_values 
        # Calcular el término dentro del exponente
        exponent_term = (n_values * log_x.unsqueeze(-1)) - torch.cumsum(log_n_values, dim=0) - 2 * log_n_k_values
        # Calcular la exponencial del término
        exp_term = torch.exp(exponent_term)
        # Multiplicar por (-1)^n y sumar sobre todos los valores de n
        sum = sum_0 + torch.sum(neg_one_power_n * exp_term, dim=-1)
        #return (neg_one_power_n * exp_term)
        return sum

def algun_nan(k):
    print(torch.sum((torch.isnan(k)).type(torch.int8)))

class CRPS_CSGDloss(nn.Module):
    def __init__(self, mean_regularitazion = True, return_mean = True):
        super(CRPS_CSGDloss, self).__init__()
        self.mean_reg = mean_regularitazion
        self.return_mean = return_mean
        #print(f'Regularization with mean: {mean_regularitazion}')
            

    def forward(self,output_notclamped, target):  #shift >= 0, scale >0, shape > 0              mean > 0, std_dev > 0
        """
        Parameters:
            output_notclamped: (N,3,...)
            target: (N, ...)
            with N the batch_size.
        """
        #Fórmula de 'Censored and shifted gamma distribution based EMOS model for probabilistic quantitative precipitation forecasting'
        output = torch.clamp(output_notclamped, min = 1e-10)
        mean, std_dev_no_reg, shift = output[:,0],output[:,1],output[:,2] #output shape: [N,C,H,W]

        if self.mean_reg:
            std_dev_regularized = std_dev_no_reg + torch.sqrt(mean) #+ 0.35 * mean
        else:
            std_dev_regularized = std_dev_no_reg
        #Obtener shape, scale y shift
        shape = mean**2 / std_dev_regularized**2
        scale = std_dev_regularized**2 / mean

        target_shifted = target + shift
        cdf_gamma_shift = self.csgd_cdf(+shift,shape,scale)

        sum1 = target_shifted * ( 2*self.csgd_cdf(target_shifted, shape, scale) - 1)
        sum2 = - (shape * scale / torch.tensor(torch.pi)) * self.beta_f(torch.tensor(0.5), shape + torch.tensor(0.5)) * (1 - self.csgd_cdf(+2 * shift, shape = 2 * shape, scale = scale))
        sum3 = scale * shape * (1 + 2*cdf_gamma_shift*self.csgd_cdf(+shift,shape + 1, scale) - cdf_gamma_shift**2 - 2*self.csgd_cdf(target_shifted,shape + 1, scale))
        sum4 = -shift * cdf_gamma_shift**2
        loss = sum1 + sum2 + sum3 + sum4

        if self.return_mean:
            return loss.mean()
        else:
            return loss
    
    def beta_f(self,x, y):
        return torch.exp(torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y))
    
    def csgd_cdf(self, input, shape, scale):  #Técnicamente le falta el desplazamiento '&' para considerarse CSGD. Sin embargo, esta '&' se da en el input en cierta manera.
        #input = torch.where(input < 0, torch.tensor(0.0), input)  #Censored
        cdf_gamma = CDFGamma.apply(shape,input/scale)
        return cdf_gamma


def set_seed(seed = 310):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.mps.deterministic = True
    torch.backends.cuda.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def num_trainable_params(net):
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return trainable_params


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


def train_model(net, tr_loader, data, mask_centers_tr, optimizer, device,
                lr_scheduler = None, loss_func = "crps", weights = None, bin_edges = None):
        net.train()
        train_loss, train_samples = 0, 0

        if loss_func == 'crps':
            criterion = CRPS_CSGDloss(weights = weights, bin_edges = bin_edges)
        elif loss_func == 'mse':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        
        for batch_idx,data_batch in enumerate(tr_loader):  #frame ==> [N, C, H, W] , targets ==> [N, N_stations]
            if data.return_hour:
                frame,targets,hours = data_batch
            else:
                frame,targets = data_batch
            optimizer.zero_grad()  
            # Mask of non-missingvalues of meteorological stations data.
            no_na_mask = ~torch.isnan(targets) 
            # Adding to the mask only the stations in the train set. Dimensions of operation: [N_stations] AND [N, N_stations] -> [N, N_stations] 
            mask_tr_ctrs = np.logical_and(mask_centers_tr, no_na_mask).bool() #Comprobado
            #With torch.nonzero we obtain the indices in the batch dimension and in the dimensions where values are True. For example: batch_tr_positions=[0,0,0,0,1,1,1...] and indices_tr_mask_stations=[2,3,6,7,2,3,7...]
            batch_tr_positions, indices_tr_mask_stations = torch.nonzero(mask_tr_ctrs, as_tuple = True) 
            num_points_to_train = len(batch_tr_positions) 
            
            x_y_tr = data.meteo_centers_info['x_y_d04'][indices_tr_mask_stations] #Is broadcastable
            x_tr_positions, y_tr_postions = x_y_tr[...,0], x_y_tr[...,1]
    
            targets_tr_ctrs = targets[mask_tr_ctrs]

            frame = frame.to(device)
            targets_tr_ctrs = targets_tr_ctrs.to(device)
            
            output = net(frame,hours) if data.return_hour else net(frame)
            # We extract from model output the positions corresponding to the meteorological stations
            output_tr_ctrs = output[batch_tr_positions, :, y_tr_postions - data.crop_y[0], x_tr_positions - data.crop_x[0] ] 
            loss = criterion(output_tr_ctrs, targets_tr_ctrs)
            loss.backward()
            optimizer.step()

            train_samples += num_points_to_train
            train_loss_batch = loss.item() * num_points_to_train
            train_loss += train_loss_batch
            if batch_idx % 5 == 0: 
                print(f' Batch {batch_idx}:{train_loss_batch}')    
                
        train_loss /= train_samples
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        return train_loss

# Same funcionality than 'train_model' but it isnt used an additional mask to only select stations tagged as train.
def train_model_all_stations(net, tr_loader, data, optimizer, device, lr_scheduler = None, loss_func = "crps", weights = None, bin_edges = None):
        net.train()
        train_loss, train_samples = 0, 0

        if loss_func == 'crps':
            criterion = CRPS_CSGDloss(weights = weights, bin_edges = bin_edges)
        elif loss_func == 'mse':
            criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        
        for batch_idx, data_batch in enumerate(tr_loader):  #frame ==> [N, C, H, W] , targets ==> [N, N_stations]
            if data.return_hour:
                frame,targets,hours = data_batch
            else:
                frame,targets = data_batch
            optimizer.zero_grad()  
            #Hacemos el filtro de las centrales meteorológicas que no tienen nan.
            no_na_mask = ~torch.isnan(targets) 
            #Con el torch.nonzero obtenemos las posiciones en la dimensión del batch y en el de las estaciones donde hay True. p.e: batch_tr_positions=[0,0,0,0,1,1,1...] y indices_tr_mask_stations=[2,3,6,7,2,3,7...]
            batch_tr_positions, indices_tr_mask_stations = torch.nonzero(no_na_mask, as_tuple = True) 
            num_points_to_train = len(batch_tr_positions) 
            
            x_y_tr = data.meteo_centers_info['x_y_d04'][indices_tr_mask_stations] #Comprobado. Aunque se van repitiendo índices sucesivamente, es broadcastable
            x_tr_positions, y_tr_postions = x_y_tr[...,0], x_y_tr[...,1]
    
            targets_tr_ctrs = targets[no_na_mask]

            frame = frame.to(device)
            targets_tr_ctrs = targets_tr_ctrs.to(device)
            
            output = net(frame,hours) if data.return_hour else net(frame)
            output_tr_ctrs = output[batch_tr_positions, :, y_tr_postions - data.crop_y[0], x_tr_positions - data.crop_x[0] ]
            loss = criterion(output_tr_ctrs, targets_tr_ctrs)
            loss.backward()
            optimizer.step()

            train_samples += num_points_to_train
            train_loss_batch = loss.item() * num_points_to_train
            train_loss += train_loss_batch
            if batch_idx % 5 == 0: 
                print(f' Batch {batch_idx}:{train_loss_batch}')       

        train_loss /= train_samples
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        return train_loss


# Test model in (all available/ masked) stations
def test_model(net, data, device , mask_centers_tst = None, tst_loader = None, per_hour = False, return_mean = True):
    if tst_loader is None:
        tst_loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, drop_last = False)
    net.eval()
    calculate_crps = CRPS_CSGDloss(return_mean = return_mean)
    if not per_hour:
        if return_mean:
            test_loss, test_bs_loss, test_mae_loss, test_mse_loss, test_samples = 0, 0, 0, 0, 0
        else:
            test_loss_list, test_bs_list, test_mae_list, test_mse_list = [], [], [], []
        
        with torch.no_grad():
            for data_batch in tst_loader:  #frame ==> [N, C, H, W] , targets ==> [N, N_stations] 
                if data.return_hour:
                    frame,targets,hours = data_batch
                else:
                    frame,targets = data_batch
                #Hacemos el filtro de las centrales meteorológicas que tienen nan
                no_na_mask = ~torch.isnan(targets) 
                if mask_centers_tst is not None:
                    no_na_mask = np.logical_and(mask_centers_tst, no_na_mask).bool()
                #Con el torch.nonzero obtenemos las posiciones en la dimensión del batch y en el de las estaciones donde hay True. p.e: batch_tr_positions=[0,0,0,0,1,1,1...] y indices_tr_mask_stations=[2,3,6,7,2,3,7...]
                batch_tst_positions, indices_test_mask_stations = torch.nonzero(no_na_mask, as_tuple = True) 
                num_points_to_test = len(batch_tst_positions) 
                
                x_y_tst = data.meteo_centers_info['x_y_d04'][indices_test_mask_stations] #Comprobado. Aunque se van repitiendo índices sucesivamente, es broadcastable
                x_tst_positions, y_tst_postions = x_y_tst[...,0], x_y_tst[...,1]

                targets_tst_ctrs = targets[no_na_mask]

                frame = frame.to(device)
                targets_tst_ctrs = targets_tst_ctrs.to(device)
                
                output = net(frame,hours) if data.return_hour else net(frame)  
                output_tst_ctrs = output[batch_tst_positions, :, y_tst_postions - data.crop_y[0], x_tst_positions - data.crop_x[0] ]

                loss = calculate_crps(output_tst_ctrs, targets_tst_ctrs)
                bs_loss = calculate_brier_score(output_tst_ctrs, targets_tst_ctrs, return_mean = return_mean)#calculate_rmse(output_tst_ctrs, targets_tst_ctrs)
                mse_loss, mae_loss = calculate_mse(output_tst_ctrs, targets_tst_ctrs, and_mae= True, return_mean = return_mean)

                if return_mean:
                    test_samples += num_points_to_test
                    test_loss_batch = loss.item() * num_points_to_test
                    test_mae_loss_batch = mae_loss.item() * num_points_to_test
                    test_mse_loss_batch = mse_loss.item() * num_points_to_test
                    test_bs_loss_batch = bs_loss.item() * num_points_to_test

                    test_loss += test_loss_batch
                    test_mae_loss += test_mae_loss_batch
                    test_bs_loss += test_bs_loss_batch
                    test_mse_loss += test_mse_loss_batch
                else:
                    test_loss_list.extend(list(loss))
                    test_mae_list.extend(list(mae_loss))
                    test_mse_list.extend(list(mse_loss))
                    test_bs_list.extend(list(bs_loss))

        if return_mean:
            test_loss /= test_samples
            test_mae_loss /= test_samples
            test_mse_loss /= test_samples
            test_bs_loss /= test_samples
            return test_loss, test_mae_loss, test_mse_loss, test_bs_loss
        else:
            return test_loss_list, test_mae_list, test_mse_list, test_bs_list
    if per_hour:
        print('WARNING: Assert that batch size of the dataloader is 1.')
        test_loss = defaultdict(float)
        nsamples_loss = defaultdict(int)
        test_mae_loss = defaultdict(float)
        test_mse_loss = defaultdict(float)
        test_bs_loss = defaultdict(float)
        with torch.no_grad():
            for (frame, targets, hours) in tst_loader:  #frame ==> [N, C, H, W] , targets ==> [N, N_stations] 
                #Hacemos el filtro de las centrales meteorológicas que tienen nan
                no_na_mask = ~torch.isnan(targets) 
                if mask_centers_tst is not None:
                    no_na_mask = np.logical_and(mask_centers_tst, no_na_mask).bool()
                #Con el torch.nonzero obtenemos las posiciones en la dimensión del batch y en el de las estaciones donde hay True. p.e: batch_tr_positions=[0,0,0,0,1,1,1...] y indices_tr_mask_stations=[2,3,6,7,2,3,7...]
                batch_tst_positions, indices_test_mask_stations = torch.nonzero(no_na_mask, as_tuple = True) 
                num_points_to_test = len(batch_tst_positions) 
                
                x_y_tst = data.meteo_centers_info['x_y_d04'][indices_test_mask_stations] #Comprobado. Aunque se van repitiendo índices sucesivamente, es broadcastable
                x_tst_positions, y_tst_postions = x_y_tst[...,0], x_y_tst[...,1]

                targets_tst_ctrs = targets[no_na_mask]

                frame = frame.to(device)
                targets_tst_ctrs = targets_tst_ctrs.to(device)
                
                output = net(frame)    
                output_tst_ctrs = output[batch_tst_positions, :, y_tst_postions - data.crop_y[0], x_tst_positions - data.crop_x[0] ]
                
                loss = calculate_crps(output_tst_ctrs, targets_tst_ctrs)
                mse_loss,mae_loss = calculate_mse(output_tst_ctrs, targets_tst_ctrs)
                bs_loss = calculate_brier_score(output_tst_ctrs, targets_tst_ctrs)

                test_loss_batch = loss.item() * num_points_to_test
                test_mae_loss_batch = mae_loss.item() * num_points_to_test
                test_mse_loss_batch = mse_loss.item() * num_points_to_test
                test_bs_loss_batch = bs_loss.item() * num_points_to_test

                hour = hours[0].item()
                test_loss[hour] += test_loss_batch
                test_mae_loss[hour] += test_mae_loss_batch
                test_mse_loss[hour] += test_mse_loss_batch
                test_bs_loss[hour] += test_bs_loss_batch
                nsamples_loss[hour] += num_points_to_test
        
        # Respectively, CRPS, MAE,MSE, BS and Number of samples
        return test_loss, test_mae_loss, test_mse_loss, test_bs_loss, nsamples_loss 
    
def obtain_params_results(net, data, device, tst_loader = None,mask_centers_tst = None):
    """
    Return fitted parameters of the results. Also returns the precipitation ground truth.
    """
    if tst_loader is None:
        tst_loader = torch.utils.data.DataLoader(data, batch_size = 32, shuffle = False, drop_last = False)
    net.eval()
    center = []
    mean = []
    std_dev_no_reg = []
    shift = []
    ground_truth = []
    with torch.no_grad():
        for data_batch in tst_loader:  #frame ==> [N, C, H, W] , targets ==> [N, N_stations] 
            if data.return_hour:
                frame,targets,hours = data_batch
            else:
                frame,targets = data_batch
            #Hacemos el filtro de las centrales meteorológicas que tienen nan
            no_na_mask = ~torch.isnan(targets) 
            if mask_centers_tst is not None:
                no_na_mask = np.logical_and(mask_centers_tst, no_na_mask).bool()
            #Con el torch.nonzero obtenemos las posiciones en la dimensión del batch y en el de las estaciones donde hay True. p.e: batch_tr_positions=[0,0,0,0,1,1,1...] y indices_tr_mask_stations=[2,3,6,7,2,3,7...]
            batch_tst_positions, indices_test_mask_stations = torch.nonzero(no_na_mask, as_tuple = True) 
            
            x_y_tst = data.meteo_centers_info['x_y_d04'][indices_test_mask_stations] #Comprobado. Aunque se van repitiendo índices sucesivamente, es broadcastable
            name_centers_batch = data.meteo_centers_info['name'][indices_test_mask_stations] #Comprobado. Aunque se van repitiendo índices sucesivamente, es broadcastable
            x_tst_positions, y_tst_postions = x_y_tst[...,0], x_y_tst[...,1]
            
            targets_tst_ctrs = targets[no_na_mask]

            frame = frame.to(device)
            targets_tst_ctrs = targets_tst_ctrs.to(device)

            output = net(frame)
            output_tst_ctrs = output[batch_tst_positions, :, y_tst_postions - data.crop_y[0], x_tst_positions - data.crop_x[0] ]
            mean_batch, std_dev_no_reg_batch, shift_batch = output_tst_ctrs[:,0],output_tst_ctrs[:,1],output_tst_ctrs[:,2]

            mean.extend(mean_batch)
            std_dev_no_reg.extend(std_dev_no_reg_batch)
            shift.extend(shift_batch)
            center.extend(name_centers_batch)
            ground_truth.extend(targets_tst_ctrs)
    
    return {'mean': mean, 'std_dev_no_reg' : std_dev_no_reg, 'shift' : shift, 'center' : center, 'prec(mm)' : ground_truth}
