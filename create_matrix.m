
function [mat,T,LAT]=create_matrix(stations,ts,component,refsys,minLat,maxLat,resolution,nbDays)


    minLat = double(minLat);
    maxLat = double(maxLat);
    resolution = double(resolution);
    nbDays = double(nbDays);

    te = ts;
    te(3) = te(3)+2;
    load station.mat

  %create binning & grid
    dt=resolution;                 %binsize time
    dlat=1;                       %binsize lat
    t=0:dt:24;
    t = t(:);
    Lat = maxLat:-dlat:minLat;
    Lat = Lat(:);
    
    [T,LAT]=meshgrid(t,Lat);
    ntm=length(T(:));
      
    mat = ones(size(T))*-1; %output matrix
 
    
    localtime = [];
    latitude = [];
    comp = [];
    maxLT = 0;                          %choose the next midnight
    
    
%     % find the  station with the latest local time at UTC timestamp
%     maxLT = 0;
%  for i = 1:length(stations)
%      stat = get_station_index(stations(i,:));
%      [st,time] = indices_alpha(ts,ts,stations(i,:));
%      lon = station(stat).longitude;
%      LT = mod(time(:,4)+time(:,5)./60+lon/15+240,24);
%      %gwa(i) = LT;
%      maxLT = max(LT,maxLT);
%  end
    
    
 for i = 1:length(stations)
     %generate data which has to be binned
     %stat = get_station_index(stations(i,:));
     stat = stations(i);
     [st,time] = indices_alpha(ts,te,station(stat).code);
     
     if isa(st, 'struct')
         if refsys=='g'
            lon = station(stat).longitude;
            lat = station(stat).latitude.*ones(1440,1);

            LT = mod(time(:,4)+time(:,5)./60+lon/15+240,24);
         elseif refsys =='m'
             ind = find(station(stat).annee == ts(1));
             lon = station(stat).lon_cgm(ind);
             [lonmag_ref,~,~,~,~]=mlt_ref(time);

             LT = rem((lon-lonmag_ref)/15+240,24);
             lat = station(stat).lat_cgm(ind).*ones(1440,1);
         else
            fprintf('Please use g or m as reference system input!') 
         end

          inds = find(abs(LT-maxLT)<0.01|abs(LT-maxLT)>23.99);
          if isempty(inds)
             fprintf('uh oh. Please turn up precision for timing...') 
          end

          localtime = [localtime;LT(inds(1):inds(1)+1440-1)];
          latitude = [latitude;lat];
          comp = [comp;st.(component)(inds(1):inds(1)+1440-1)];
     end      
 end

%bin the 3vectors comp;localtime;latitude  
%caution: for stations with same latitudes & local time the mean is generated
for itps=1:ntm
  ind=find(abs(localtime-T(itps))<=dt/2&abs(latitude-LAT(itps))<=dlat/2);
    if ~isempty(ind) 
        tmp = comp(ind);
        nnans = find(~isnan(tmp));
        mat(itps) = mean(tmp(nnans));
    else
        mat(itps) = NaN;
    end
end
      
% plot it

    if 0
        ytick_pos = [];
         for i = 1:length(stations)
            stat = stations(i);%get_station_index(stations(i,:));

            if refsys=='g'
                ytick_pos = [ytick_pos,station(stat).latitude];
             elseif refsys =='m'
                ind = find(station(stat).annee == ts(1));
                ytick_pos = [ytick_pos,station(stat).lat_cgm(ind)];
             else
                fprintf('Please use g or m as reference system input!') 
             end

         end

         [ytick_pos,inds]=sort(ytick_pos);
         %ytick_labels = station(stations(inds)).code;%stations(inds,:);
         stats = stations(inds);
         ytick_labels = {station(stats).code};


        figure      
        image(T(:),LAT(:),mat,'CDataMapping','scaled','AlphaData',~isnan(mat))
        set(gca,'YDir','normal')
        colorbar
        colormap(jet(15))
        yticks(ytick_pos)
        yticklabels(ytick_labels)
        txt = sprintf('%s %4i-%02i-%02i %s',component,ts(1),ts(2),ts(3),refsys);
        title(txt)
    end  
end
 


% refsys = 'm';
% component = 'y2';
% 
%   
%  stations = ['irt';'mmb';'bmt';'lzh';'kny'];
%  stations = ['nvs';'irt';'bmt';'lzh';'kak';'kny'];
%  ts = [2010 01 13 00 00 00];
%  te = [2010 01 15 23 59 00];
 