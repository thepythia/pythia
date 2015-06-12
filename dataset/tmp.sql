create table tmp_haose_show as 
select t.auction_id,
    t.pict_url,
    t.cat1,
    t.color_type
from (
	select a.cat1, e.web_id as color_type,
		   concat(e.web_id, "_", a.cat1) as rowkey,
		   a.auction_id, 
		   c.pict_url,
	      row_number() over (partition by a.cat1, e.web_id 
						  order by a.cat1, e.web_id, b.item_score desc, a.fg_percent desc) as rank
	from (select pt as cat1, 
	            auction_id, fg_segid, fg_percent, pict_url
		  from tbai_haose_auctions_colors
		  where bg_segid>=326
		        and bg_percent>=0.20
		        and fg_percent>=0.1) a
	join (select auction_id, max(item_score) as item_score
		  from wl_ind.superuser_auction_score
		  where ds=20150106 group by auction_id) b
	on (a.auction_id=b.auction_id)  
	join (select auction_id, pict_url    --to eliminate changed auction/picture
         from tbai_image_auctions_online
         where pt=20150106 and pict_url is not null and length(pict_url)>5) c
    on (a.auction_id=c.auction_id and a.pict_url=c.pict_url)
	join (select seg_id, web_id from tbai_haose_45color_to_hsbsegid) e
	on (a.fg_segid=e.seg_id)
    ) t
where t.rank <= 2;