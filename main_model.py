import numpy as np
import torch
import torch.nn as nn
from diff_models import Guide_diff
from transformer import *
from layers import *


class Model1(nn.Module):
    def __init__(self, config, target_dim, device, inputdim=2):
        super().__init__()
        self.config = config
        self.channels = config['diffusion']['channels']
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config['diffusion']['num_steps'],
            embedding_dim=config['diffusion']['diffusion_embedding_dim'])
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        is_unconditional = config["model"]["is_unconditional"]
        target_strategy = config["model"]["target_strategy"]
        self.device = "cuda:1"
        self.target_dim = target_dim
        self.seq_len = 24

        self.adj = config["diffusion"]["adj_file"]
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        embed_layer = nn.Embedding(num_embeddings=target_dim, embedding_dim=self.emb_feature_dim)
        self.device = device
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        # config_diff["side_dim"] = self.emb_total_dim
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["diffusion"]["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion"]["diffusion_embedding_dim"],
                    nheads=config["diffusion"]["nheads"],
                    target_dim=target_dim,
                    proj_t=config["diffusion"]["proj_t"],
                    is_adp=config["diffusion"]["is_adp"],
                    device=config_diff["device"],
                    adj_file=config["diffusion"]["adj_file"],
                    is_cross_t=config["diffusion"]["is_cross_t"],
                    is_cross_s=config["diffusion"]["is_cross_s"],
                )
                for _ in range(config["diffusion"]["layers"])
            ]
        )



    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):

            current_sample = torch.zeros_like(observed_data)

            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            else:
                diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
            predicted = self.diffmodel(diff_input, 0)


            current_sample = predicted

            # if t > 0:
            #     noise = torch.randn_like(current_sample)
            #     sigma = (
            #         (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
            #     ) ** 0.5
            #     current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples



    def process_data(self, config, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2).to(self.device)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        ones = torch.zeros_like(noisy_data)
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)

        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * ones).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = coeffs.unsqueeze(1)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask

    def diffmodel(self, x1):
        x3 = x1['observed_data'].to(self.device).float().transpose(-1, -2)
        x = x1['observed_data'].to(self.device).float().transpose(-1, -2)
        cond_mask = x1['gt_mask'].to(self.device).float().transpose(-1, -2)
        ob_mask = x1["observed_mask"].to(self.device).float().transpose(-1, -2)
        x2 = x * cond_mask

        side_info = cond_mask
        itp_x = cond_mask * x

        observed_tp = x1["timepoints"]

        if self.adj == 'AQI36':
            self.adj1 = get_adj_AQI36()
        elif self.adj == 'la':
            self.adj1 = get_similarity_metrla(thr=0.1)
        elif self.adj == 'bay':
            self.adj1 = torch.randn(325,325).uniform_(0., 0.1)
        self.support = compute_support_gwn(self.adj1, device=self.device)
        node_num = self.adj1.shape[0]
        self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
        self.support.append([self.nodevec1, self.nodevec2])
        side_info = self.get_side_info(observed_tp, cond_mask)
        total_input = self.set_input_to_diffmodel(x2, x, cond_mask)
        B, inputdim, K, L = total_input.shape

        x = total_input

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)


        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, side_info, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)

        return x

    def forward(self, x1, state):
        x3 = x1['observed_data'].to(self.device).float().transpose(-1, -2)
        x = x1['observed_data'].to(self.device).float().transpose(-1, -2)
        cond_mask = x1['gt_mask'].to(self.device).float().transpose(-1, -2)
        ob_mask = x1["observed_mask"].to(self.device).float().transpose(-1, -2)
        x2 = x * cond_mask
        x = self.diffmodel(x1)

        if state == 0:
            observed_data, target_mask, observed_mask = self.evaluate(x, 25)

        # a[t] = self.r[0]
            return x, observed_data, target_mask, observed_mask
        elif state == 1:
            return self.calc_loss(x3, cond_mask, ob_mask, x)
        else:
            return self.calc_loss_valid(x3, cond_mask, ob_mask, x)

    def calc_loss_valid(
    self, observed_data, cond_mask, observed_mask, predicted
    ):

          # calculate loss for all t
        loss = self.calc_loss(
            observed_data, cond_mask, observed_mask
        )

        return loss / self.num_steps

    def calc_loss(
    self, observed_data, cond_mask, observed_mask, predicted
    ):
        B, K, L = observed_data.shape



        target_mask = observed_mask - cond_mask
        residual = (observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = residual.sum() / (num_eval if num_eval > 0 else 1)
        return loss


class Score(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.seq_len = seq_len

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.use_guide = config["model"]["use_guide"]

        self.cde_output_channels = config["diffusion"]["channels"]
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["device"] = device
        self.device = device

        input_dim = 2
        nheads = 8
        channels = 64
        self.diffmodel = Guide_diff(config_diff, input_dim, target_dim, self.use_guide)
        self.update_gate = MessagePN2(c_in=24, c_out=24, heads=nheads, layers=1, channels=channels)
        order = 2
        include_self = True
        proj_t = 64
        device = None
        is_adp = False
        adj_file = None
        is_cross_t = False
        is_cross_s = True
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)
        # parameters for diffusion models
        # self.num_steps = config_diff["num_steps"]
        # if config_diff["schedule"] == "quad":
        #     self.beta = np.linspace(
        #         config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
        #     ) ** 2
        # elif config_diff["schedule"] == "linear":
        #     self.beta = np.linspace(
        #         config_diff["beta_start"], config_diff["beta_end"], self.num_steps
        #     )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info




    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train
    ):
        loss_sum = 0
          # calculate loss for all t
        loss = self.calc_loss(
            observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=1
        )
        loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.zeros_like(observed_data)
        noisy_data =  observed_data
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        if not self.use_guide:
            itp_info = cond_mask * observed_data
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)

        target_mask = observed_mask - cond_mask
        residual = torch.abs(observed_data - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = residual.sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        ones = torch.zeros_like(noisy_data)
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)

        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * ones).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.zeros_like(observed_data)


            if self.is_unconditional == True:
                diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
            else:
                if not self.use_guide:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                else:
                    diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), itp_info, cond_mask)


            current_sample = predicted

            # if t > 0:
            #     noise = torch.randn_like(current_sample)
            #     sigma = (
            #         (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
            #     ) ** 0.5
            #     current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        C = channel
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        attn = torch.zeros_like(y)
        # y = y.reshape(B, channel, K, L).reshape(B * channel, K, L)
        # y = torch.cat([self.time_shift(y)[:, :L, :C // 2], y[:, :L, C // 2:]], dim=2).transpose(1, 2) # 只需增加这句

        y, attn = self.update_gate(y, base_shape)
        # y = self.tcn(y)
        # y = self.impgconv(y, base_shape)

        # y = y.reshape(B, channel, K, L).reshape(B, channel, K*L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

        return y

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        if self.use_guide:
            itp_info = coeffs.unsqueeze(1)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(observed_data, cond_mask, observed_mask, side_info, itp_info, is_train)
        return output

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = coeffs.unsqueeze(1)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp



class Score_aqi36(Score):
    def __init__(self, config, device, target_dim=36, seq_len=36):
        super(Score_aqi36, self).__init__(target_dim, seq_len, config, device)
        self.config = config



    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()

        cond_mask = get_randmask(observed_mask)
        # cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


class Score_MetrLA(Model1):
    def __init__(self, config, device, target_dim=207, seq_len=24):
        super(Score_MetrLA, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


class Score_PemsBAY(Score):
    def __init__(self, config, device, target_dim=325, seq_len=24):
        super(Score_PemsBAY, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        # cut_length = batch["cut_length"].to(self.device).long()
        coeffs = None
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = get_randmask(observed_mask)
        cond_mask = cond_mask.float()
        cut_length = torch.zeros(len(observed_data)).to(self.device).long()
        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )

def get_randmask(observed_mask):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    for i in range(len(observed_mask)):
        sample_ratio = np.random.rand()
        num_observed = observed_mask[i].sum().item()
        num_masked = round(num_observed * sample_ratio)
        rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask



class MessagePN2(nn.Module):
    def __init__(self, c_in, c_out, heads, layers, channels):
        super(MessagePN2, self).__init__()
        self.c_in = c_in
        self.c_tmp = 2 * c_in

        self.attn = MultiHeadAttention(heads, 3, self.c_in, 0.1)
        # self.linear = nn.Linear(self.c_in, 2 * self.c_in)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_in, 6 * (self.c_tmp + self.c_in)),  ###
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(6 * (self.c_tmp + self.c_in), c_out),
            nn.ReLU(inplace=False),
        )
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)

    def forward(self, data, base_shape):
        # context = context.reshape(base_shape)
        # context = self.SRU(context)
        # context = context.reshape(B * channel, K, L)

        out, attn = self.attn(data, data, data, base_shape)
        out1 = torch.cat((out, data), dim=-1)
        # out = self.linear(data)

        return self.mlp(out1), attn


#
# class ResidualBlock1(nn.Module):
#     def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
#         super().__init__()
#         self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
#         self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
#         self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
#         self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
#
#         self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
#         self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
#
#     def forward_time(self, y, base_shape):
#         B, channel, K, L = base_shape
#         if L == 1:
#             return y
#         y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
#         y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
#         y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
#         return y
#
#     def forward_feature(self, y, base_shape):
#         B, channel, K, L = base_shape
#         if K == 1:
#             return y
#         y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
#         y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
#         y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
#         return y
#
#     def forward(self, x):
#         B, channel, K, L = x.shape
#         base_shape = x.shape
#         x = x.reshape(B, channel, K * L)
#         y = self.forward_time(x, base_shape)
#         y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
#         y = self.mid_projection(y)  # (B,2*channel,K*L)
#
#
#         gate, filter = torch.chunk(y, 2, dim=1)
#         y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
#         y = self.output_projection(y)
#
#         residual, skip = torch.chunk(y, 2, dim=1)
#         x = x.reshape(base_shape)
#         residual = residual.reshape(base_shape)
#         skip = skip.reshape(base_shape)
#         return (x + residual) / math.sqrt(2.0), skip
#


class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        # self.forward_time = TemporalLearning(channels=channels, nheads=nheads, is_cross=is_cross_t)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)
        self.update_gate = MessagePN2(c_in=24, c_out=24, heads=nheads, layers=1, channels=channels)

        adj1 = torch.randn(325,325).to(device)
        adj1.require_grad = True

    def forward(self, x, side_info, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        # diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x

        y = self.forward_time(y, base_shape)
        # y = self.tcn(y)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        # y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        C = channel
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        attn = torch.zeros_like(y)
        # y = y.reshape(B, channel, K, L).reshape(B * channel, K, L)
        # y = torch.cat([self.time_shift(y)[:, :L, :C // 2], y[:, :L, C // 2:]], dim=2).transpose(1, 2) # 只需增加这句

        y, attn = self.update_gate(y, base_shape)
        # y = self.tcn(y)
        # y = self.impgconv(y, base_shape)

        # y = y.reshape(B, channel, K, L).reshape(B, channel, K*L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

        return y

    #
    # def forward_feature(self, y, base_shape):
    #     B, channel, K, L = base_shape
    #     if L == 1:
    #         return y
    #     C = channel
    #     y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * L, channel, K)
    #     attn = torch.zeros_like(y)
    #     # y = y.reshape(B, channel, K, L).reshape(B * channel, K, L)
    #     # y = torch.cat([self.time_shift(y)[:, :L, :C // 2], y[:, :L, C // 2:]], dim=2).transpose(1, 2) # 只需增加这句
    #
    #     y, attn = self.spa_gate(y, base_shape)
    #     # y = self.tcn(y)
    #     # y = self.impgconv(y, base_shape)
    #
    #     # y = y.reshape(B, channel, K, L).reshape(B, channel, K*L)
    #     y = y.reshape(B, L, channel, K).permute(0, 3, 1, 2).reshape(B, channel, K * L)
    #
    #     return y
    # #



class MessagePN2(nn.Module):
    def __init__(self, c_in, c_out, heads, layers, channels):
        super(MessagePN2, self).__init__()
        self.c_in = c_in
        self.c_tmp = 2 * c_in

        self.attn = MultiHeadAttention(heads, 3, self.c_in, 0.1)
        # self.linear = nn.Linear(self.c_in, 2 * self.c_in)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_in, 6 * (self.c_tmp + self.c_in)),  ###
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(6 * (self.c_tmp + self.c_in), c_out),
            nn.ReLU(inplace=False),
        )
        self.SRU = SRU(64,
                       group_num=4,
                       gate_treshold=0.5)

    def forward(self, data, base_shape):
        # context = context.reshape(base_shape)
        # context = self.SRU(context)
        # context = context.reshape(B * channel, K, L)

        out, attn = self.attn(data, data, data, base_shape)
        out1 = torch.cat((out, data), dim=-1)
        # out = self.linear(data)

        return self.mlp(out1), attn