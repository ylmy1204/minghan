clc; clear; close all;
totalStart = tic;

%% ==================== 点云加载 ====================
[f1,p1] = uigetfile({'*.ply;*.pcd;*.txt'},'选择源点云');
if isequal(f1,0), error('未选择'); end
original = pcread(fullfile(p1,f1));
fprintf('原始点云点数: %d\n', original.Count);

%% ==================== 实验配置 ====================
N_trials    = 30;
seeds       = 1:N_trials;
rot_max_deg = 45;
trans_ratio = 0.15;
n_methods   = 9;
method_names = {
    'P2P-ICP', ...
    'P2Plane-ICP', ...
    'NDT', ...
    'FPFH+RANSAC+P2P', ...
    'FPFH+RANSAC+P2Plane', ...
    'FPFH+GNC+P2P', ...
    'FGR+P2Plane', ...
    'TLS+P2Plane', ...
    'Ours'};

ALL_RRE  = zeros(n_methods, N_trials);
ALL_RTE  = zeros(n_methods, N_trials);
ALL_RMSE = zeros(n_methods, N_trials);
ALL_Time = zeros(n_methods, N_trials);
ALL_OK   = false(n_methods, N_trials);

% 每个 trial 保存完整结果用于可视化选取
TRIAL_T    = cell(n_methods, N_trials);
TRIAL_src  = cell(N_trials, 1);
TRIAL_tgt  = cell(N_trials, 1);
TRIAL_T_gt = cell(N_trials, 1);
TRIAL_sf   = zeros(N_trials, 1);

%% ==================== 主循环 ====================
for trial = 1:N_trials
    fprintf('\n======== Trial %d / %d ========\n', trial, N_trials);
    rng(seeds(trial));

    %% 随机扰动
    diagLen = sqrt(diff(original.XLimits)^2 + ...
                   diff(original.YLimits)^2 + ...
                   diff(original.ZLimits)^2);
    ax_v = randn(3,1); ax_v = ax_v/norm(ax_v);
    ang  = deg2rad((rand()*2-1)*rot_max_deg);
    K    = [0,-ax_v(3),ax_v(2); ax_v(3),0,-ax_v(1); -ax_v(2),ax_v(1),0];
    R_n  = eye(3)+sin(ang)*K+(1-cos(ang))*(K*K);
    t_n  = (rand(3,1)*2-1)*trans_ratio*diagLen;
    T_noise = eye(4); T_noise(1:3,1:3)=R_n; T_noise(1:3,4)=t_n;
    T_gt    = inv(T_noise);
    source  = pctransform(original, affine3d(T_noise'));
    target  = original;

    %% 自适应体素
    diagT = sqrt(diff(target.XLimits)^2+diff(target.YLimits)^2+diff(target.ZLimits)^2);
    diagS = sqrt(diff(source.XLimits)^2+diff(source.YLimits)^2+diff(source.ZLimits)^2);
    sf    = max(diagT, diagS);
    nPts  = max(target.Count, source.Count);
    [voxC, voxM, voxF] = adaptiveVoxels(sf, nPts);
    kN    = 30;

    %% 配置
    cfg = struct();
    cfg.icp_max_iter  = 100;
    cfg.icp_tol       = 1e-5;
    cfg.ransac_iter   = 3000;
    cfg.ransac_thr    = voxC * 1.5;
    cfg.gnc_max_iter  = 100;
    cfg.gnc_mu_factor = 1.4;
    cfg.kNormal       = kN;
    cfg.scaleFactor   = sf;
    cfg.voxCoarse     = voxC;
    cfg.voxMid        = voxM;
    cfg.voxFine       = voxF;

        %% ===== 计时外只保留：下采样点云 =====
    pcTC_C = preprocessPC(target, voxC, kN);
    pcSC_C = preprocessPC(source, voxC, kN);
    pcTC_M = preprocessPC(target, voxM, kN);
    pcSM_M = preprocessPC(source, voxM, kN);
    pcTC_F = preprocessPC(target, voxF, kN);
    pcSF_F = preprocessPC(source, voxF, kN);

    % 特征和匹配点提前算好，供各方法共享（计时外）
    featV = voxC*3; searchR = featV*2;
    sKey  = pcdownsample(pcSC_C,'gridAverage',featV);
    tKey  = pcdownsample(pcTC_C,'gridAverage',featV);
    sDesc = computeSimpleDescriptor(sKey,pcSC_C,searchR);
    tDesc = computeSimpleDescriptor(tKey,pcTC_C,searchR);
    [mSrc,mTgt] = mutualMatchRatio(sDesc,tDesc,sKey,tKey,0.85);
    if size(mSrc,1)<15
        [mSrc,mTgt] = mutualMatchRatio(sDesc,tDesc,sKey,tKey,0.95);
    end
    if size(mSrc,1)<6
        [mSrc,mTgt,~] = mutualMatch(sDesc,tDesc,sKey,tKey);
    end
    fprintf('  匹配点数: %d\n', size(mSrc,1));

    %% M1: P2P-ICP
    t_s = tic;
    T_rans_m1 = ransacRigid(mSrc, mTgt, cfg);   % RANSAC计入
    src_t = pointCloud((T_rans_m1(1:3,1:3)*pcSC_C.Location'+T_rans_m1(1:3,4))');
    try
        tf = pcregistericp(src_t,pcTC_C,'Metric','pointToPoint',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{1} = safeTransform(double(tf.T)'*T_rans_m1,T_rans_m1,sf);
    catch; T_results{1}=T_rans_m1; end
    ALL_Time(1,trial) = toc(t_s);

    %% M2: P2Plane-ICP
    t_s = tic;
    T_rans_m2 = ransacRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_rans_m2(1:3,1:3)*pcSC_C.Location'+T_rans_m2(1:3,4))');
    src_t = preprocessPC(src_t, voxC, kN);
    warning('off','all');
    try
        tf = pcregistericp(src_t,pcTC_C,'Metric','pointToPlane',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{2} = safeTransform(double(tf.T)'*T_rans_m2,T_rans_m2,sf);
    catch; T_results{2}=T_rans_m2; end
    warning('on','all');
    ALL_Time(2,trial) = toc(t_s);

    %% M3: NDT
    t_s = tic;
    T_rans_m3 = ransacRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_rans_m3(1:3,1:3)*pcSC_C.Location'+T_rans_m3(1:3,4))');
    try
        tf = pcregisterndt(src_t,pcTC_C,voxC*2,...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{3} = safeTransform(double(tf.T)'*T_rans_m3,T_rans_m3,sf);
    catch; T_results{3}=T_rans_m3; end
    ALL_Time(3,trial) = toc(t_s);

    %% M4: FPFH+RANSAC+P2P
    t_s = tic;
    T_rans_m4 = ransacRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_rans_m4(1:3,1:3)*pcSM_M.Location'+T_rans_m4(1:3,4))');
    try
        tf = pcregistericp(src_t,pcTC_M,'Metric','pointToPoint',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{4} = safeTransform(double(tf.T)'*T_rans_m4,T_rans_m4,sf);
    catch; T_results{4}=T_rans_m4; end
    ALL_Time(4,trial) = toc(t_s);

    %% M5: FPFH+RANSAC+P2Plane
    t_s = tic;
    T_rans_m5 = ransacRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_rans_m5(1:3,1:3)*pcSM_M.Location'+T_rans_m5(1:3,4))');
    src_t = preprocessPC(src_t, voxM, kN);
    warning('off','all');
    try
        tf = pcregistericp(src_t,pcTC_M,'Metric','pointToPlane',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{5} = safeTransform(double(tf.T)'*T_rans_m5,T_rans_m5,sf);
    catch; T_results{5}=T_rans_m5; end
    warning('on','all');
    ALL_Time(5,trial) = toc(t_s);

    %% M6: FPFH+GNC+P2P
    t_s = tic;
    T_gnc_m6 = gncRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_gnc_m6(1:3,1:3)*pcSM_M.Location'+T_gnc_m6(1:3,4))');
    try
        tf = pcregistericp(src_t,pcTC_M,'Metric','pointToPoint',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{6} = safeTransform(double(tf.T)'*T_gnc_m6,T_gnc_m6,sf);
    catch; T_results{6}=T_gnc_m6; end
    ALL_Time(6,trial) = toc(t_s);

    %% M7: FGR+P2Plane
    t_s = tic;
    T_fgr_m7 = fgrRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_fgr_m7(1:3,1:3)*pcSM_M.Location'+T_fgr_m7(1:3,4))');
    src_t = preprocessPC(src_t, voxM, kN);
    warning('off','all');
    try
        tf = pcregistericp(src_t,pcTC_M,'Metric','pointToPlane',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{7} = safeTransform(double(tf.T)'*T_fgr_m7,T_fgr_m7,sf);
    catch; T_results{7}=T_fgr_m7; end
    warning('on','all');
    ALL_Time(7,trial) = toc(t_s);

    %% M8: TLS+P2Plane
    t_s = tic;
    T_tls_m8 = tlsRigid(mSrc, mTgt, cfg);
    src_t = pointCloud((T_tls_m8(1:3,1:3)*pcSM_M.Location'+T_tls_m8(1:3,4))');
    src_t = preprocessPC(src_t, voxM, kN);
    warning('off','all');
    try
        tf = pcregistericp(src_t,pcTC_M,'Metric','pointToPlane',...
            'MaxIterations',cfg.icp_max_iter,'Tolerance',[cfg.icp_tol,cfg.icp_tol]);
        T_results{8} = safeTransform(double(tf.T)'*T_tls_m8,T_tls_m8,sf);
    catch; T_results{8}=T_tls_m8; end
    warning('on','all');
    ALL_Time(8,trial) = toc(t_s);

    %% Ours
    t_s = tic;
    T_r = ransacRigid(mSrc, mTgt, cfg);
    T_g = gncRigid(mSrc, mTgt, cfg);
    T_f = fgrRigid(mSrc, mTgt, cfg);
    T_t = tlsRigid(mSrc, mTgt, cfg);
    T_init_ours = robustMultiStartCoarse(mSrc, mTgt, cfg, ...
        pcSC_C, pcTC_C, T_r, T_g, T_f, T_t);
    T_results{9} = oursMethod(pcSC_C,pcTC_C,pcSM_M,pcTC_M, ...
        pcSF_F,pcTC_F,T_init_ours,cfg);
    ALL_Time(9,trial) = toc(t_s);

   
    %% 评估
    srcPtsEval = pcSF_F.Location;
    rre_thr = 5.0; rte_thr = sf*0.05;
    for m = 1:n_methods
        [rre,rte,rmse] = evaluateRegistration(T_results{m},T_gt,srcPtsEval);
        ALL_RRE(m,trial)  = rre;
        ALL_RTE(m,trial)  = rte;
        ALL_RMSE(m,trial) = rmse;
        ALL_OK(m,trial)   = (rre<rre_thr)&&(rte<rte_thr);
    end
    fprintf('  RRE(Ours)=%.3f°  RTE(Ours)=%.4f  SR=%d%%\n',...
        ALL_RRE(9,trial),ALL_RTE(9,trial),round(100*sum(ALL_OK(9,1:trial))/trial));

      %% 保存本次 trial 的完整信息
    for m = 1:n_methods
        TRIAL_T{m,trial} = T_results{m};
    end
    TRIAL_src{trial}  = source;
    TRIAL_tgt{trial}  = target;
    TRIAL_T_gt{trial} = T_gt;
    TRIAL_sf(trial)   = sf;
end

%% ==================== 统计输出 ====================
mean_RRE  = mean(ALL_RRE,2);  std_RRE  = std(ALL_RRE,0,2);
mean_RTE  = mean(ALL_RTE,2);  std_RTE  = std(ALL_RTE,0,2);
mean_RMSE = mean(ALL_RMSE,2); std_RMSE = std(ALL_RMSE,0,2);
mean_Time = mean(ALL_Time,2); std_Time = std(ALL_Time,0,2);
SR        = sum(ALL_OK,2)/N_trials*100;

full_names = {
    'M1: P2P-ICP', 'M2: P2Plane-ICP', 'M3: NDT', ...
    'M4: FPFH+RANSAC+P2P', 'M5: FPFH+RANSAC+P2Plane', ...
    'M6: FPFH+GNC+P2P', 'M7: FGR+P2Plane', 'M8: TLS+P2Plane', ...
    'Ours: Adaptive+GNC+WP2Plane'};

short_names = {
    'P2P-ICP','P2Plane-ICP','NDT',
    'RANSAC+P2P','RANSAC+P2Plane','GNC+P2P',
    'FGR+P2Plane','TLS+P2Plane','Ours'};

fprintf('\n========== %d 次实验统计 ==========\n', N_trials);
fprintf('%-38s %17s %17s %17s %14s %7s\n',...
    '方法','RRE°(mean±std)','RTE(mean±std)','RMSE(mean±std)','Time(s)','SR%%');
fprintf('%s\n',repmat('-',1,114));
for m = 1:n_methods
    fprintf('%-38s %7.4f±%-8.4f %7.4f±%-8.4f %7.4f±%-8.4f %5.2f±%-5.2f %6.1f%%\n',...
        full_names{m},mean_RRE(m),std_RRE(m),mean_RTE(m),std_RTE(m),...
        mean_RMSE(m),std_RMSE(m),mean_Time(m),std_Time(m),SR(m));
end
fprintf('%s\n',repmat('-',1,114));

%% ==================== 可视化 trial 选取 ====================
rre_thr_vis = 5.0;
vis_trial = zeros(n_methods,1);
for m = 1:n_methods
    fail_idx = find(~ALL_OK(m,:));
    if ~isempty(fail_idx)
        [~,worst]    = max(ALL_RRE(m, fail_idx));
        vis_trial(m) = fail_idx(worst);
    else
        vis_trial(m) = N_trials;
    end
end

fprintf('\n可视化 trial 选取：\n');
for m = 1:n_methods
    t = vis_trial(m);
    if ALL_OK(m,t), tag='成功(全部成功)'; else, tag='失败'; end
    fprintf('  %-28s → Trial %2d  [%s]  RRE=%.3f°\n',...
        short_names{m}, t, tag, ALL_RRE(m,t));
end

%% ==================== 配准结果可视化（完整点云）====================
fprintf('\n正在生成可视化...\n');

%% ==================== 配准结果可视化 ====================
fig_reg = figure('Name','配准结果对比','Color','k','Position',[20,20,1920,1080]);

for m = 1:n_methods
    t       = vis_trial(m);
    T_m     = TRIAL_T{m,t};
    T_gt_v  = TRIAL_T_gt{t};
    sf_v    = TRIAL_sf(t);
    src_pts = TRIAL_src{t}.Location;
    tgt_pts = TRIAL_tgt{t}.Location;

    
    max_disp = 5000;
    idx_t = randperm(size(tgt_pts,1), min(max_disp,size(tgt_pts,1)));
    idx_s = randperm(size(src_pts,1), min(max_disp,size(src_pts,1)));
    tgt_d = tgt_pts(idx_t,:);
    pts_a = (T_m(1:3,1:3)*src_pts' + T_m(1:3,4))';
    src_d = pts_a(idx_s,:);

    ax = subplot(3,3,m);
    scatter3(ax, tgt_d(:,1),tgt_d(:,2),tgt_d(:,3), ...
        1, [0,0.82,0.82], 'filled');
    hold(ax,'on');
    scatter3(ax, src_d(:,1),src_d(:,2),src_d(:,3), ...
        1, [1,0.55,0], 'filled');
    hold(ax,'off');

    [rre_v,rte_v,~] = evaluateRegistration(T_m, T_gt_v, src_pts);
    ok_v = (rre_v < rre_thr_vis) && (rte_v < sf_v*0.05);
    if ok_v, tc=[0.2,1.0,0.2]; mk='OK';
    else,    tc=[1.0,0.3,0.3]; mk='FAIL'; end

    set(ax,'Color',[0.04,0.04,0.04],...
           'XColor',[0.5,0.5,0.5],'YColor',[0.5,0.5,0.5],...
           'ZColor',[0.5,0.5,0.5],'FontSize',7.5,...
           'GridColor',[0.25,0.25,0.25],'GridAlpha',0.35);
    grid(ax,'on'); view(ax,3);
    xlabel(ax,'X','Color',[0.5,0.5,0.5],'FontSize',7);
    ylabel(ax,'Y','Color',[0.5,0.5,0.5],'FontSize',7);
    zlabel(ax,'Z','Color',[0.5,0.5,0.5],'FontSize',7);
    title(ax, sprintf('%s  (Trial %d)\n[%s]  RRE=%.3f  RTE=%.3f',...
        short_names{m}, t, mk, rre_v, rte_v),...
        'Color',tc,'FontSize',9,'FontWeight','bold','Interpreter','none');
end

sgtitle(fig_reg,...
    '各方法配准结果  青=目标  橙=配准后源  有失败时展示最差失败trial',...
    'Color','w','FontSize',11,'FontWeight','bold');


%% ==================== 消融实验 ====================
fprintf('\n========== 消融实验 ==========\n');
N_ab    = 5;
ab_seed = 42;
n_ab    = 9;
ab_names = {'A1-Full','A2-w/o MS','A3-w/o Adap','A4-RANSAC init',...
            'A5-w/o RK','A6-w/o NC','A7-w/o DW','A8-P2P','A9-w/o ES'};

AB_RRE  = zeros(n_ab,N_ab);
AB_RTE  = zeros(n_ab,N_ab);
AB_RMSE = zeros(n_ab,N_ab);
AB_Time = zeros(n_ab,N_ab);
AB_OK   = false(n_ab,N_ab);

for ab_t = 1:N_ab
    rng(ab_seed + ab_t - 1);
    diagLen_ab = sqrt(diff(original.XLimits)^2+diff(original.YLimits)^2+diff(original.ZLimits)^2);
    ax_ab=randn(3,1); ax_ab=ax_ab/norm(ax_ab);
    ang_ab=deg2rad((rand()*2-1)*rot_max_deg);
    K_ab=[0,-ax_ab(3),ax_ab(2);ax_ab(3),0,-ax_ab(1);-ax_ab(2),ax_ab(1),0];
    R_ab=eye(3)+sin(ang_ab)*K_ab+(1-cos(ang_ab))*(K_ab*K_ab);
    t_ab=(rand(3,1)*2-1)*trans_ratio*diagLen_ab;
    T_n_ab=eye(4); T_n_ab(1:3,1:3)=R_ab; T_n_ab(1:3,4)=t_ab;
    T_gt_ab=inv(T_n_ab);
    src_ab=pctransform(original,affine3d(T_n_ab'));
    tgt_ab=original;

    diagT_ab=sqrt(diff(tgt_ab.XLimits)^2+diff(tgt_ab.YLimits)^2+diff(tgt_ab.ZLimits)^2);
    sf_ab=max(diagT_ab,...
        sqrt(diff(src_ab.XLimits)^2+diff(src_ab.YLimits)^2+diff(src_ab.ZLimits)^2));
    [vC_ab,vM_ab,vF_ab]=adaptiveVoxels(sf_ab,max(tgt_ab.Count,src_ab.Count));

    cfg_ab=struct('icp_max_iter',100,'icp_tol',1e-5,'ransac_iter',3000,...
        'ransac_thr',vC_ab*1.5,'gnc_max_iter',100,'gnc_mu_factor',1.4,...
        'kNormal',30,'scaleFactor',sf_ab,'voxCoarse',vC_ab,'voxMid',vM_ab,'voxFine',vF_ab);

    pcTC_ab=preprocessPC(tgt_ab,vC_ab,30); pcSC_ab=preprocessPC(src_ab,vC_ab,30);
    pcTM_ab=preprocessPC(tgt_ab,vM_ab,30); pcSM_ab=preprocessPC(src_ab,vM_ab,30);
    pcTF_ab=preprocessPC(tgt_ab,vF_ab,30); pcSF_ab=preprocessPC(src_ab,vF_ab,30);

    featV_ab=vC_ab*3; searchR_ab=featV_ab*2;
    sKey_ab=pcdownsample(pcSC_ab,'gridAverage',featV_ab);
    tKey_ab=pcdownsample(pcTC_ab,'gridAverage',featV_ab);
    sDesc_ab=computeSimpleDescriptor(sKey_ab,pcSC_ab,searchR_ab);
    tDesc_ab=computeSimpleDescriptor(tKey_ab,pcTC_ab,searchR_ab);
    [mS_ab,mT_ab]=mutualMatchRatio(sDesc_ab,tDesc_ab,sKey_ab,tKey_ab,0.85);
    if size(mS_ab,1)<6,[mS_ab,mT_ab,~]=mutualMatch(sDesc_ab,tDesc_ab,sKey_ab,tKey_ab);end

    T_rans_ab=ransacRigid(mS_ab,mT_ab,cfg_ab);
    T_gnc_ab=gncRigid(mS_ab,mT_ab,cfg_ab);
    T_fgr_ab=fgrRigid(mS_ab,mT_ab,cfg_ab);
    T_tls_ab=tlsRigid(mS_ab,mT_ab,cfg_ab);
    T_init_ab=robustMultiStartCoarse(mS_ab,mT_ab,cfg_ab,...
                   pcSC_ab,pcTC_ab,T_rans_ab,T_gnc_ab,T_fgr_ab,T_tls_ab);

    srcE_ab=pcSF_ab.Location;
    rre_thr_ab=5.0; rte_thr_ab=sf_ab*0.05;
    T_ab=cell(n_ab,1);

    % A1: 完整Ours
    t_s=tic;
    T_ab{1}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,pcSF_ab,pcTF_ab,T_init_ab,cfg_ab);
    AB_Time(1,ab_t)=toc(t_s);

    % A2: 无多尺度（只用coarse层）
    t_s=tic;
    cfg2=cfg_ab;
    src_c2=applyTransform(pcSC_ab,T_init_ab);
    [Td2,~]=weightedP2PlaneICP(src_c2,pcTC_ab,cfg2,'coarse');
    T_ab{2}=safeTransform(Td2*T_init_ab,T_init_ab,sf_ab);
    AB_Time(2,ab_t)=toc(t_s);

    % A3: 无自适应体素（固定体素）
    t_s=tic;
    cfg3=cfg_ab; fixed_vox=sf_ab*0.01;
    cfg3.voxCoarse=fixed_vox; cfg3.voxMid=fixed_vox*0.6; cfg3.voxFine=fixed_vox*0.3;
    pcSC3=preprocessPC(src_ab,fixed_vox,30);
    pcTC3=preprocessPC(tgt_ab,fixed_vox,30);
    pcSM3=preprocessPC(src_ab,fixed_vox*0.6,30);
    pcTM3=preprocessPC(tgt_ab,fixed_vox*0.6,30);
    pcSF3=preprocessPC(src_ab,fixed_vox*0.3,30);
    pcTF3=preprocessPC(tgt_ab,fixed_vox*0.3,30);
    T_ab{3}=oursMethod(pcSC3,pcTC3,pcSM3,pcTM3,pcSF3,pcTF3,T_init_ab,cfg3);
    AB_Time(3,ab_t)=toc(t_s);

    % A4: 用RANSAC初始化（替换鲁棒多起点）
    t_s=tic;
    T_ab{4}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_rans_ab,cfg_ab);
    AB_Time(4,ab_t)=toc(t_s);

    % A5: 无鲁棒核
    t_s=tic;
    cfg5=cfg_ab; cfg5.no_robust_kernel=true;
    T_ab{5}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_init_ab,cfg5);
    AB_Time(5,ab_t)=toc(t_s);

    % A6: 无法向约束
    t_s=tic;
    cfg6=cfg_ab; cfg6.no_normal_constraint=true;
    T_ab{6}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_init_ab,cfg6);
    AB_Time(6,ab_t)=toc(t_s);

    % A7: 无距离权重
    t_s=tic;
    cfg7=cfg_ab; cfg7.no_dist_weight=true;
    T_ab{7}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_init_ab,cfg7);
    AB_Time(7,ab_t)=toc(t_s);

    % A8: P2P替换P2Plane
    t_s=tic;
    cfg8=cfg_ab; cfg8.use_p2p=true;
    T_ab{8}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_init_ab,cfg8);
    AB_Time(8,ab_t)=toc(t_s);

    % A9: 无早停
    t_s=tic;
    cfg9=cfg_ab; cfg9.no_early_stop=true;
    T_ab{9}=oursMethod(pcSC_ab,pcTC_ab,pcSM_ab,pcTM_ab,...
                       pcSF_ab,pcTF_ab,T_init_ab,cfg9);
    AB_Time(9,ab_t)=toc(t_s);

    for a=1:n_ab
        [rre_a,rte_a,rmse_a]=evaluateRegistration(T_ab{a},T_gt_ab,srcE_ab);
        AB_RRE(a,ab_t)=rre_a; AB_RTE(a,ab_t)=rte_a; AB_RMSE(a,ab_t)=rmse_a;
        AB_OK(a,ab_t)=(rre_a<rre_thr_ab)&&(rte_a<rte_thr_ab);
    end
end

% 消融统计输出
fprintf('%-20s %15s %15s %15s %12s %7s\n',...
    '消融变体','RRE°(mean±std)','RTE(mean±std)','RMSE(mean±std)','Time(s)','SR%%');
fprintf('%s\n',repmat('-',1,88));
for a=1:n_ab
    fprintf('%-20s %6.4f±%-7.4f %6.4f±%-7.4f %6.4f±%-7.4f %4.2f±%-4.2f %6.1f%%\n',...
        ab_names{a},...
        mean(AB_RRE(a,:)),std(AB_RRE(a,:)),...
        mean(AB_RTE(a,:)),std(AB_RTE(a,:)),...
        mean(AB_RMSE(a,:)),std(AB_RMSE(a,:)),...
        mean(AB_Time(a,:)),std(AB_Time(a,:)),...
        sum(AB_OK(a,:))/N_ab*100);
end
fprintf('%s\n',repmat('-',1,88));
fprintf('总耗时: %.1f 秒\n', toc(totalStart));


%% ============================================================
%%                     辅助函数
%% 1. 自适应体素
function [voxC,voxM,voxF] = adaptiveVoxels(sf,nPts)
    rC=max(0.006,min(0.025,(8000/max(nPts,1))^(1/3)));
    rF=max(0.001,min(0.008,(30000/max(nPts,1))^(1/3)));
    rM=(rC+rF)/2;
    voxC=sf*rC; voxM=sf*rM; voxF=sf*rF;
end

%% 2. 预处理+法向估计
function pc_out = preprocessPC(pc_in,voxSize,kNormal)
    pc_ds=pcdownsample(pc_in,'gridAverage',voxSize);
    pts=pc_ds.Location; n=size(pts,1);
    normals=zeros(n,3);
    ns=KDTreeSearcher(pts);
    cg=mean(pts,1);
    for i=1:n
        idx=knnsearch(ns,pts(i,:),'K',min(kNormal,n));
        nb=pts(idx,:); mu=mean(nb,1);
        C=(nb-mu)'*(nb-mu);
        [V,D]=eig(C); [~,im]=min(diag(D));
        nv=V(:,im)';
        if dot(nv,pts(i,:)-cg)<0, nv=-nv; end
        normals(i,:)=nv;
    end
    pc_out=pointCloud(pts,'Normal',normals);
end

%% 3. 简单几何描述子
function desc = computeSimpleDescriptor(keyPC,densePC,searchRad)
    nk=size(keyPC.Location,1); desc=zeros(nk,8);
    ns=KDTreeSearcher(densePC.Location);
    for i=1:nk
        idx=rangesearch(ns,keyPC.Location(i,:),searchRad); idx=idx{1};
        if numel(idx)<4
            idx=knnsearch(ns,keyPC.Location(i,:),'K',min(10,densePC.Count));
        end
        nb=densePC.Location(idx,:); mu=mean(nb,1);
        C=(nb-mu)'*(nb-mu)/max(size(nb,1)-1,1);
        ev=sort(eig(C),'descend'); ev=max(ev,0); s=sum(ev)+1e-10;
        desc(i,1)=(ev(1)-ev(2))/s; desc(i,2)=(ev(2)-ev(3))/s;
        desc(i,3)=ev(3)/s;         desc(i,4)=ev(3)/(ev(1)+1e-10);
        if ~isempty(densePC.Normal)
            nm=densePC.Normal(idx,:); mn=mean(nm,1);
            desc(i,5)=mn(1); desc(i,6)=mn(2); desc(i,7)=mn(3);
            desc(i,8)=mean(std(nm,0,1));
        end
    end
    mu_d=mean(desc,1); sd_d=std(desc,0,1)+1e-10;
    desc=(desc-mu_d)./sd_d;
end

%% 4. 双向互匹配
function [mSrc,mTgt,scores] = mutualMatch(srcDesc,tgtDesc,srcKey,tgtKey)
    ns_t=KDTreeSearcher(tgtDesc); ns_s=KDTreeSearcher(srcDesc);
    [idx_st,d_st]=knnsearch(ns_t,srcDesc,'K',1);
    [idx_ts,~   ]=knnsearch(ns_s,tgtDesc,'K',1);
    mSrc=[]; mTgt=[]; scores=[];
    for i=1:size(srcDesc,1)
        j=idx_st(i);
        if idx_ts(j)==i
            mSrc(end+1,:)=srcKey.Location(i,:);
            mTgt(end+1,:)=tgtKey.Location(j,:);
            scores(end+1)=d_st(i);
        end
    end
end

%% 5. 比率测试匹配
function [mSrc,mTgt] = mutualMatchRatio(srcDesc,tgtDesc,srcKey,tgtKey,thr)
    ns_t=KDTreeSearcher(tgtDesc); ns_s=KDTreeSearcher(srcDesc);
    K2t=min(2,size(tgtDesc,1)); K2s=min(2,size(srcDesc,1));
    [idx_st,d_st]=knnsearch(ns_t,srcDesc,'K',K2t);
    [idx_ts,d_ts]=knnsearch(ns_s,tgtDesc,'K',K2s);
    mSrc=[]; mTgt=[];
    for i=1:size(srcDesc,1)
        if K2t>=2 && d_st(i,1)/(d_st(i,2)+1e-10)>=thr, continue; end
        j=idx_st(i,1);
        if K2s>=2 && d_ts(j,1)/(d_ts(j,2)+1e-10)>=thr, continue; end
        if idx_ts(j,1)==i
            mSrc(end+1,:)=srcKey.Location(i,:);
            mTgt(end+1,:)=tgtKey.Location(j,:);
        end
    end
end

%% 6. GNC
function T = gncRigid(mSrc,mTgt,cfg)
    if size(mSrc,1)<3, T=eye(4); return; end
    n=size(mSrc,1); w=ones(n,1);
    res2=sum((mSrc-mTgt).^2,2);
    mu=max(res2)/(max(max(res2)/(median(res2)+1e-10)-1,1e-4));
    R=eye(3); t=zeros(3,1);
    for it=1:cfg.gnc_max_iter
        sw=sum(w); if sw<1e-10,break; end
        cS=sum(w.*mSrc,1)/sw; cT=sum(w.*mTgt,1)/sw;
        H=((mSrc-cS).*w)'*(mTgt-cT);
        [U,~,V]=svd(H); R=V*diag([1,1,det(V*U')])*U'; t=cT'-R*cS';
        res2=sum(((mSrc*R'+t')-mTgt).^2,2);
        sc=median(sqrt(res2))+1e-10; mu=mu*cfg.gnc_mu_factor;
        w=(mu./(mu+res2/sc^2)).^2;
    end
    inl=w>0.5;
    if sum(inl)>=3,[R,t]=svdRigid(mSrc(inl,:),mTgt(inl,:));end
    T=eye(4); T(1:3,1:3)=R; T(1:3,4)=t;
end

%% 7. RANSAC
function T = ransacRigid(mSrc,mTgt,cfg)
    n=size(mSrc,1); if n<3,T=eye(4);return;end
    best_inl=0; T=eye(4); thr2=cfg.ransac_thr^2;
    for it=1:cfg.ransac_iter
        idx=randperm(n,3);
        try
            [R,t]=svdRigid(mSrc(idx,:),mTgt(idx,:));
            pts_t=(R*mSrc'+t)';
            inl=sum((pts_t-mTgt).^2,2)<thr2;
            if sum(inl)>best_inl
                best_inl=sum(inl);
                if sum(inl)>=3
                    [Rr,tr]=svdRigid(mSrc(inl,:),mTgt(inl,:));
                    T(1:3,1:3)=Rr; T(1:3,4)=tr;
                end
            end
        catch;end
    end
end

%% 8. FGR
function T = fgrRigid(mSrc,mTgt,cfg)
    if size(mSrc,1)<3,T=eye(4);return;end
    mu=1.0; n=size(mSrc,1); l=ones(n,1); R=eye(3); t=zeros(3,1);
    for it=1:64
        w=l.^2; [R,t]=svdRigidWeighted(mSrc,mTgt,w);
        pts_t=(R*mSrc'+t)'; res2=sum((pts_t-mTgt).^2,2);
        l=mu./(mu+res2); mu=mu/1.4; if mu<1e-7,break;end
    end
    inl=l>0.5;
    if sum(inl)>=3,[R,t]=svdRigid(mSrc(inl,:),mTgt(inl,:));end
    T=eye(4); T(1:3,1:3)=R; T(1:3,4)=t;
end

%% 9. TLS
function T = tlsRigid(mSrc,mTgt,cfg)
    if size(mSrc,1)<3,T=eye(4);return;end
    src=mSrc; tgt=mTgt; T=eye(4);
    for it=1:20
        if size(src,1)<3,break;end
        [R,t]=svdRigid(src,tgt);
        pts_t=(R*src'+t)'; res2=sum((pts_t-tgt).^2,2);
        thr=median(res2)*2.5+1e-10; inl=res2<thr;
        if sum(inl)<3||sum(inl)==size(src,1)
            T(1:3,1:3)=R; T(1:3,4)=t; break;
        end
        src=src(inl,:); tgt=tgt(inl,:);
        T(1:3,1:3)=R; T(1:3,4)=t;
    end
end

%% 10. 鲁棒多起点粗配准（解决匹配点少/大量误匹配）
function T_best = robustMultiStartCoarse(mSrc,mTgt,cfg,...
                      pcSC,pcTC,T_rans,T_gnc,T_fgr,T_tls)
    sf=cfg.scaleFactor;

    % 候选1-4：描述子匹配得到的粗配准
    candidates={T_rans,T_gnc,T_fgr,T_tls};

    % 候选5-12：在coarse层用ICP验证每个候选，选RMSE最小的
    ns_c=KDTreeSearcher(pcTC.Location);
    best_rmse=inf; T_best=T_rans;

    for k=1:numel(candidates)
        Tc=candidates{k};
        if ~isValidTransform(Tc,sf), continue; end
        % 用ICP跑少量迭代（20次）验证
        cfg_quick=cfg; cfg_quick.icp_max_iter=20; cfg_quick.no_early_stop=false;
        src_t=applyTransform(pcSC,Tc);
        try
            [Td,~]=weightedP2PlaneICP(src_t,pcTC,cfg_quick,'coarse');
            T_test=Td*Tc;
        catch
            T_test=Tc;
        end
        if ~isValidTransform(T_test,sf), T_test=Tc; end
        % 计算RMSE（采样500点）
        pts_t=(T_test(1:3,1:3)*pcSC.Location'+T_test(1:3,4))';
        nSamp=min(500,size(pts_t,1));
        idx_s=randperm(size(pts_t,1),nSamp);
        [~,d]=knnsearch(ns_c,pts_t(idx_s,:),'K',1);
        rmse_k=sqrt(mean(d.^2));
        if rmse_k<best_rmse
            best_rmse=rmse_k; T_best=T_test;
        end
    end

    % 如果最优RMSE仍然很大，补充随机多起点ICP
    rmse_threshold=cfg.voxCoarse*5;
    if best_rmse>rmse_threshold
        nExtra=8;
        for k=1:nExtra
            % 随机旋转初始化
            ax_r=randn(3,1); ax_r=ax_r/norm(ax_r);
            ang_r=deg2rad(rand()*360);
            Kr=[0,-ax_r(3),ax_r(2);ax_r(3),0,-ax_r(1);-ax_r(2),ax_r(1),0];
            Rr=eye(3)+sin(ang_r)*Kr+(1-cos(ang_r))*(Kr*Kr);
            % 平移：对齐质心
            cS=mean(pcSC.Location,1)'; cT=mean(pcTC.Location,1)';
            tr=cT-Rr*cS;
            T_rand=eye(4); T_rand(1:3,1:3)=Rr; T_rand(1:3,4)=tr;

            cfg_quick=cfg; cfg_quick.icp_max_iter=20;
            src_r=applyTransform(pcSC,T_rand);
            try
                [Td,~]=weightedP2PlaneICP(src_r,pcTC,cfg_quick,'coarse');
                T_test=Td*T_rand;
            catch
                T_test=T_rand;
            end
            if ~isValidTransform(T_test,sf), continue; end
            pts_t=(T_test(1:3,1:3)*pcSC.Location'+T_test(1:3,4))';
            nSamp=min(500,size(pts_t,1));
            idx_s=randperm(size(pts_t,1),nSamp);
            [~,d]=knnsearch(ns_c,pts_t(idx_s,:),'K',1);
            rmse_k=sqrt(mean(d.^2));
            if rmse_k<best_rmse
                best_rmse=rmse_k; T_best=T_test;
            end
        end
    end
end

%% 11. Ours主方法
function T_final = oursMethod(pcSC,pcTC,pcSM,pcTM,pcSF,pcTF,T_init,cfg)
    sf=cfg.scaleFactor;
    src_c=applyTransform(pcSC,T_init);
    [Tdc,~]=weightedP2PlaneICP(src_c,pcTC,cfg,'coarse');
    T_c=Tdc*T_init; if ~isValidTransform(T_c,sf),T_c=T_init;end

    src_m=applyTransform(pcSM,T_c);
    [Tdm,~]=weightedP2PlaneICP(src_m,pcTM,cfg,'mid');
    T_m=Tdm*T_c; if ~isValidTransform(T_m,sf),T_m=T_c;end

    src_f=applyTransform(pcSF,T_m);
    [Tdf,~]=weightedP2PlaneICP(src_f,pcTF,cfg,'fine');
    T_f=Tdf*T_m;
    T_final=safeTransform(T_f,T_m,sf);
end

%% 12. 加权点面ICP
function [T_delta,converged] = weightedP2PlaneICP(srcPC,tgtPC,cfg,level)
    sf=cfg.scaleFactor;
    switch level
        case 'coarse', max_dist=cfg.voxCoarse*3; max_iter=cfg.icp_max_iter; cauchy_c=cfg.voxCoarse*2;
        case 'mid',    max_dist=cfg.voxMid*3;    max_iter=cfg.icp_max_iter; cauchy_c=cfg.voxMid*2;
        case 'fine',   max_dist=cfg.voxFine*3;   max_iter=cfg.icp_max_iter; cauchy_c=cfg.voxFine*2;
        otherwise,     max_dist=sf*0.05; max_iter=cfg.icp_max_iter; cauchy_c=sf*0.03;
    end
    tol=cfg.icp_tol; col_scale=max(sf,1e-6);
    srcPts=srcPC.Location; srcNrm=srcPC.Normal;
    tgtPts=tgtPC.Location; tgtNrm=tgtPC.Normal;
    ns_tgt=KDTreeSearcher(tgtPts);
    R_acc=eye(3); t_acc=zeros(3,1); prev_rmse=inf; converged=false;

    use_rk = ~isfield(cfg,'no_robust_kernel')     || ~cfg.no_robust_kernel;
    use_nc = ~isfield(cfg,'no_normal_constraint') || ~cfg.no_normal_constraint;
    use_dw = ~isfield(cfg,'no_dist_weight')       || ~cfg.no_dist_weight;
    use_p2p=  isfield(cfg,'use_p2p')              &&  cfg.use_p2p;
    use_es = ~isfield(cfg,'no_early_stop')        || ~cfg.no_early_stop;

    for iter=1:max_iter
        [idx,dist]=knnsearch(ns_tgt,srcPts,'K',1);
        valid=dist<max_dist; nv=sum(valid);
        if nv<6,break;end
        sp=srcPts(valid,:); tp=tgtPts(idx(valid),:);
        tn=tgtNrm(idx(valid),:); sn=srcNrm(valid,:); d=dist(valid);

        w=ones(nv,1);
        if use_dw, sigma2=(max_dist/2)^2; w=w.*exp(-d.^2/sigma2); end
        if use_nc
            snn=sqrt(sum(sn.^2,2)); tnn=sqrt(sum(tn.^2,2));
            if sum((snn>0.5)&(tnn>0.5))>=3
                w=w.*max(abs(sum(sn.*tn,2)),0.1);
            end
        end
        res_p2pl=abs(sum((tp-sp).*tn,2));
        if use_rk, w=w./(1+(res_p2pl/(cauchy_c+1e-10)).^2); end
        if any(~isfinite(w))||sum(w)<1e-10, w=ones(nv,1); end
        w=w/(mean(w)+1e-10);

        if use_p2p
            [R_step,t_step]=svdRigidWeighted(sp,tp,w);
        else
            A=zeros(nv,6); b=zeros(nv,1);
            for i=1:nv
                nx=tn(i,1);ny=tn(i,2);nz=tn(i,3);
                px=sp(i,1);py=sp(i,2);pz=sp(i,3);
                A(i,1)=(nz*py-ny*pz)/col_scale; A(i,2)=(nx*pz-nz*px)/col_scale;
                A(i,3)=(ny*px-nx*py)/col_scale; A(i,4)=nx; A(i,5)=ny; A(i,6)=nz;
                b(i)=sum((tp(i,:)-sp(i,:)).*tn(i,:));
            end
            sw=sqrt(w); Aw=A.*sw; bw=b.*sw;
            lambda=max(1e-4*sum(Aw(:).^2)/6,1e-6);
            AtA=Aw'*Aw+lambda*eye(6);
            if rcond(AtA)<1e-12, lambda=lambda*1e3; AtA=Aw'*Aw+lambda*eye(6); end
            x=AtA\(Aw'*bw);
            if any(~isfinite(x))
                [R_step,t_step]=svdRigidWeighted(sp,tp,w);
            else
                a1=x(1)/col_scale; a2=x(2)/col_scale; a3=x(3)/col_scale;
                ang_i=norm([a1,a2,a3]);
                if ang_i<1e-10
                    R_step=eye(3);
                else
                    axv=[a1,a2,a3]/ang_i;
                    Ki=[0,-axv(3),axv(2);axv(3),0,-axv(1);-axv(2),axv(1),0];
                    R_step=eye(3)+sin(ang_i)*Ki+(1-cos(ang_i))*(Ki*Ki);
                end
                t_step=[x(4);x(5);x(6)];
            end
        end
        if ~isfinite(norm(t_step))||norm(t_step)>sf*2,break;end
        R_acc=R_step*R_acc; t_acc=R_step*t_acc+t_step;
        srcPts=(R_step*srcPts'+t_step)'; srcNrm=(R_step*srcNrm')';
        rmse_cur=sqrt(mean(sum(((R_step*sp'+t_step)'-tp).^2,2)));
        if use_es&&iter>3&&abs(prev_rmse-rmse_cur)/(prev_rmse+1e-10)<tol
            converged=true; break;
        end
        prev_rmse=rmse_cur;
    end
    T_delta=eye(4); T_delta(1:3,1:3)=R_acc; T_delta(1:3,4)=t_acc;
end

%% 13. applyTransform
function pc_out = applyTransform(pc_in,T)
    R=T(1:3,1:3); t=T(1:3,4);
    pts=(R*pc_in.Location'+t)';
    if ~isempty(pc_in.Normal)
        pc_out=pointCloud(pts,'Normal',(R*pc_in.Normal')');
    else
        pc_out=pointCloud(pts);
    end
end

%% 14. SVD等权
function [R,t] = svdRigid(src,tgt)
    cs=mean(src,1); ct=mean(tgt,1);
    [U,~,V]=svd((src-cs)'*(tgt-ct));
    R=V*diag([1,1,det(V*U')])*U'; t=ct'-R*cs';
end

%% 15. SVD加权
function [R,t] = svdRigidWeighted(src,tgt,w)
    w=w(:)/(sum(w(:))+1e-10);
    cs=sum(w.*src,1); ct=sum(w.*tgt,1);
    [U,~,V]=svd(((src-cs).*w)'*(tgt-ct));
    R=V*diag([1,1,det(V*U')])*U'; t=ct'-R*cs';
end

%% 16. 变换有效性检查
function ok = isValidTransform(T,sf)
    ok=false;
    if any(~isfinite(T(:))), return; end
    if abs(det(T(1:3,1:3))-1)>0.05, return; end
    if norm(T(1:3,4))>sf*3, return; end
    ok=true;
end

%% 17. 安全变换
function T_out = safeTransform(T_new,T_fb,sf)
    if isValidTransform(T_new,sf), T_out=T_new;
    else, T_out=T_fb; end
end

%% 18. 评估
function [rre,rte,rmse] = evaluateRegistration(T_est,T_gt,srcPts)
    R_err=T_est(1:3,1:3)/T_gt(1:3,1:3);
    rre=rad2deg(acos(max(-1,min(1,(trace(R_err)-1)/2))));
    rte=norm(T_est(1:3,4)-T_gt(1:3,4));
    pts_e=(T_est(1:3,1:3)*srcPts'+T_est(1:3,4))';
    pts_g=(T_gt(1:3,1:3) *srcPts'+T_gt(1:3,4))';
    rmse=sqrt(mean(sum((pts_e-pts_g).^2,2)));
end