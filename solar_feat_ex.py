import solar_features as sol

test = sol.sun_img('20111114175814Mh.jpg')


filaments = sol.sun_img.detecting_filaments(test)

prominences = sol.sun_img.detecting_prominences(test)

image_list = sol.sun_img.make_image_list('20120109000014', 7,10)

sol.sun_img.time_ev_filament(image_list)


