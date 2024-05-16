
(function ($) {
	
    'use strict';

	 function site_search(){
			jQuery('a[href="#search"]').on('click', function(event) {                    
			jQuery('#search').addClass('open');
			jQuery('#search > form > input[type="search"]').focus();
		});
					
		jQuery('#search, #search button.close').on('click keyup', function(event) {
			if (event.target === this || event.target.className === 'close') {
				jQuery(this).removeClass('open');
			}
		});  
	 }	
	 

	function video_responsive(){	
		jQuery('iframe[src*="youtube.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');
		jQuery('iframe[src*="vimeo.com"]').wrap('<div class="embed-responsive embed-responsive-16by9"></div>');	
	}  


	function magnific_popup(){
        jQuery('.mfp-gallery').magnificPopup({
          delegate: '.mfp-link',
          type: 'image',
          tLoading: 'Loading image #%curr%...',
          mainClass: 'mfp-img-mobile',
          gallery: {
            enabled: true,
            navigateByImgClick: true,
            preload: [0,1] 
          },
          image: {
            tError: '<a href="%url%">The image #%curr%</a> could not be loaded.',
          }
       });
	}


	function magnific_video(){	
		jQuery('.mfp-video').magnificPopup({
			type: 'iframe',
		});
	}


	function popup_vertical_center(){	
		jQuery(function() {
			function reposition() {
				var modal = jQuery(this),
				dialog = modal.find('.modal-dialog');
				modal.css('display', 'block');
				
				dialog.css("margin-top", Math.max(0, (jQuery(window).height() - dialog.height()) / 2));
			}
			
			jQuery('.modal').on('show.bs.modal', reposition);
	
			jQuery(window).on('resize', function() {
				jQuery('.modal:visible').each(reposition);
			});
		});
	}

	
	function sticky_header(){
		if(jQuery('.sticky-header').length){
			var sticky = new Waypoint.Sticky({
			  element: jQuery('.sticky-header')
			})
		}
	}


	function scroll_top(){
		jQuery("button.scroltop").on('click', function() {
			jQuery("html, body").animate({
				scrollTop: 0
			}, 1000);
			return false;
		});

		jQuery(window).on("scroll", function() {
			var scroll = jQuery(window).scrollTop();
			if (scroll > 900) {
				jQuery("button.scroltop").fadeIn(1000);
			} else {
				jQuery("button.scroltop").fadeOut(1000);
			}
		});
	}

	 
	function input_type_file_form(){
		jQuery(document).on('change', '.btn-file :file', function() {
			var input = jQuery(this),
				numFiles = input.get(0).files ? input.get(0).files.length : 1,
				label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
			input.trigger('fileselect', [numFiles, label]);
		});

		jQuery('.btn-file :file').on('fileselect', function(event, numFiles, label) {
			var input = jQuery(this).parents('.input-group').find(':text'),
				log = numFiles > 10 ? numFiles + ' files selected' : label;
			if (input.length) {
				input.val(log);
			} else {
				if (log) alert(log);
			}
		});	
	}


	function placeholderSupport(){

		jQuery.support.placeholder = ('placeholder' in document.createElement('input'));

		if (!jQuery.support.placeholder) {
			jQuery("[placeholder]").on('focus', function () {
				if (jQuery(this).val() === jQuery(this).attr("placeholder")) jQuery(this).val("");
			}).blur(function () {
				if (jQuery(this).val() === "") jQuery(this).val(jQuery(this).attr("placeholder"));
			}).blur();

			jQuery("[placeholder]").parents("form").on('submit', function () {
				jQuery(this).find('[placeholder]').each(function() {
					if (jQuery(this).val() === jQuery(this).attr("placeholder")) {
						 jQuery(this).val("");
					}
				});
			});
		}
	
	}	



	function footer_fixed() {
	  jQuery('.site-footer').css('display', 'block');
	  jQuery('.site-footer').css('height', 'auto');
	  var footerHeight = jQuery('.site-footer').outerHeight();
	  jQuery('.footer-fixed > .page-wraper').css('padding-bottom', footerHeight);
	  jQuery('.site-footer').css('height', footerHeight);
	}


	function accordion_active() {
		$('.acod-head a').on('click', function() {
			$('.acod-head').removeClass('acc-actives');
			$(this).parents('.acod-head').addClass('acc-actives');
			$('.acod-title').removeClass('acc-actives'); 
			$(this).parent().addClass('acc-actives'); 
			($(this).parents('.acod-head').attr('class'));
		 });
	}	


	
	function mobile_nav(){
		jQuery(".sub-menu, .mega-menu").parent('li').addClass('has-child');
		jQuery("<div class='fa fa-angle-right submenu-toogle'></div>").insertAfter(".has-child > a");

		jQuery('.has-child a+.submenu-toogle').on('click',function(ev) {

			jQuery(this).parent().siblings(".has-child ").children(".sub-menu, .mega-menu").slideUp(500, function(){
				jQuery(this).parent().removeClass('nav-active');
			});

			jQuery(this).next(jQuery('.sub-menu, .mega-menu ')).slideToggle(500, function(){
				jQuery(this).parent().toggleClass('nav-active');
			});

			ev.stopPropagation();
		});

	}

	
	function mobile_side_drawer(){
		jQuery('#mobile-side-drawer').on('click', function () { 
			jQuery('.mobile-sider-drawer-menu').toggleClass('active');
		});
	}


	function blog_related_slider(){
	jQuery('.blog-related-slider').owlCarousel({
		loop:true,
		margin:30,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			480:{
				items:2
			},			
			767:{
				items:2
			},
			1000:{
				items:2
			}
		}
	});
	}
	

	function aboutus_carousel(){
	jQuery('.about-us-carousel').owlCarousel({
		loop:true,
		autoplay:true,
		autoplayTimeout:3000,
		margin:30,
		nav:true,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		items:1,
		dots: false,
	});
	}
	

	function home_carousel_1(){
	jQuery('.home-carousel-1').owlCarousel({
        loop:true,
		margin:0,
		autoplay:true,
		autoplayTimeout:3000,
		//center: true,
		nav:false,
		dots: true,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			480:{
				items:1
			},			
			767:{
				items:1
			},
			1000:{
				items:1
			}
		}
	});
	}	
	

	function home_carousel_2(){
	jQuery('.home-carousel-2').owlCarousel({
        loop:true,
		autoplay:true,
		margin:0,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			480:{
				items:1
			},			
			767:{
				items:1
			},
			1000:{
				items:1
			}
		}
	});
	}
			

	function home_projects_filter(){
		
		var owl = jQuery('.owl-carousel-filter').owlCarousel({
		loop:false,
		autoplay:true,
		margin:20,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			480:{
				items:1
			},
			580:{
				items:2
			},						
			767:{
				items:2
			},
			991:{
				items:3
			},			
			1152:{
				items:4
			},
			1360:{
				items:4
			},
			1366:{
				items:5
			}	
		    }
		})
		
		

		jQuery('.btn-filter-wrap').on('click', '.btn-filter', function(e) {
			var filter_data = jQuery(this).data('filter');

		
			if(jQuery(this).hasClass('btn-active')) return;

			
			jQuery(this).addClass('btn-active').siblings().removeClass('btn-active');

		
			owl.owlFilter(filter_data, function(_owl) { 
				jQuery(_owl).find('.item').each(owlAnimateFilter); 
			});
		})
	
	
	
	}
	
	
	

	function testimonial_home(){
	jQuery('.testimonial-home').owlCarousel({
		loop:true,
		autoplay:false,
		margin:80,
		nav:false,
		dots: true,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			991:{
				items:2
			}
		}
	});
	}	
	

	function home_client_carousel(){
	jQuery('.home-client-carousel').owlCarousel({
		loop:true,
		margin:10,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:2
			},
			480:{
				items:3
			},			
			767:{
				items:4
			},
			1000:{
				items:5
			}
		}
	});
	}	



	function home_client_carousel_2(){
	jQuery('.home-client-carousel-2').owlCarousel({
		loop:true,
		margin:10,
		autoplay:true,
		nav:false,
		dots: true,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:2
			},
			480:{
				items:3
			},			
			767:{
				items:4
			},
			1000:{
				items:5
			}
		}
	});
	}		

	function work_carousel(){
	jQuery('.work-carousel').owlCarousel({
        loop:true,
		autoplay:false,
		center: true,
		items:3,
		margin:0,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			854:{
				items:2
			}
			
		}
	});
	}	



	function portfolio_nogap_carousel(){
	jQuery('.portfolio-carousel-nogap').owlCarousel({
		loop:true,
		margin:0,
		nav:true,
		dots: false,
		navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
		responsive:{
			0:{
				items:1
			},
			480:{
				items:2
			},			
			767:{
				items:3
			},
			1000:{
				items:3
			}
		}
	});
	}
	

	function hover_tab(){
		jQuery('.circle-block-outer .nav-link').hover(function() {
            jQuery(this).tab('show');
        });
	}


       



	function equalheight(container) {
		var currentTallest = 0, 
			currentRowStart = 0, 
			rowDivs = new Array(), 
			$el, topPosition = 0,
			currentDiv = 0;

		jQuery(container).each(function() {
			$el = jQuery(this);
			jQuery($el).height('auto');
			var topPostion = $el.position().top;
			if (currentRowStart != topPostion) {
				for (currentDiv = 0; currentDiv < rowDivs.length; currentDiv++) {
					rowDivs[currentDiv].height(currentTallest);
				}
				rowDivs.length = 0;
				currentRowStart = topPostion;
				currentTallest = $el.height();
				rowDivs.push($el);
			} else {

				rowDivs.push($el);
				currentTallest = (currentTallest < $el.height()) ? ($el.height()) : (currentTallest);
			}

			for (currentDiv = 0; currentDiv < rowDivs.length; currentDiv++) {
				rowDivs[currentDiv].height(currentTallest);
			}
		});
	}



	
	var TxtType = function(el, toRotate, period) {
        this.toRotate = toRotate;
        this.el = el;
        this.loopNum = 0;
        this.period = parseInt(period, 10) || 2000;
        this.txt = '';
        this.tick();
        this.isDeleting = false;
    };

    TxtType.prototype.tick = function() {
        var i = this.loopNum % this.toRotate.length;
        var fullTxt = this.toRotate[i];

        if (this.isDeleting) {
        this.txt = fullTxt.substring(0, this.txt.length - 1);
        } else {
        this.txt = fullTxt.substring(0, this.txt.length + 1);
        }

        this.el.innerHTML = '<span class="wrap">'+this.txt+'</span>';

        var that = this;
        var delta = 200 - Math.random() * 100;

        if (this.isDeleting) { delta /= 2; }

        if (!this.isDeleting && this.txt === fullTxt) {
        delta = this.period;
        this.isDeleting = true;
        } else if (this.isDeleting && this.txt === '') {
        this.isDeleting = false;
        this.loopNum++;
        delta = 500;
        }

        setTimeout(function() {
        that.tick();
        }, delta);
    };

    window.onload = function() {
        var elements = document.getElementsByClassName('typewrite');
        for (var i=0; i<elements.length; i++) {
            var toRotate = elements[i].getAttribute('data-type');
            var period = elements[i].getAttribute('data-period');
            if (toRotate) {
              new TxtType(elements[i], JSON.parse(toRotate), period);
            }
        }
        // INJECT CSS
        var css = document.createElement("style");
        css.type = "text/css";
        css.innerHTML = ".typewrite > .wrap {}";
        document.body.appendChild(css);
    };



	function animate_content(){
		jQuery('.animate').scrolla({
			mobile: false,
			once: true
		});
	}


	function masonryBox() {
        if ( jQuery().isotope ) {      
            var $container = jQuery('.portfolio-wrap');
                $container.isotope({
                    itemSelector: '.masonry-item',
                    transitionDuration: '1s',
					originLeft: true
                });

            jQuery('.masonry-filter li').on('click',function() {                           
                var selector = jQuery(this).find("a").attr('data-filter');
                jQuery('.masonry-filter li').removeClass('active');
                jQuery(this).addClass('active');
                $container.isotope({ filter: selector });
                return false;
            });
    	};
	}	


	
	function page_loader() {
		$('.loading-area').fadeOut(1000)
	};



    function color_fill_header() {
        var scroll = $(window).scrollTop();
        if(scroll >= 100) {
            $(".is-fixed").addClass("color-fill");
        } else {
            $(".is-fixed").removeClass("color-fill");
        }
    };	


	jQuery(document).ready(function() {
			
		site_search(),
		
		video_responsive(),
		
		magnific_popup(),
		
		magnific_video(),
		
		popup_vertical_center();
			
		sticky_header(),
		
		scroll_top(),
		 	
		input_type_file_form(),
			
		placeholderSupport(),
		
		footer_fixed(),
				
		accordion_active(),
		
		mobile_nav(),
		
	    mobile_side_drawer(),
		
		blog_related_slider(),
		aboutus_carousel(),

		home_carousel_1(),

		home_carousel_2(),

		home_projects_filter(),
	
		testimonial_home(),
	
		home_client_carousel(),

		home_client_carousel_2(),		

		hover_tab(),

	        portfolio_nogap_carousel()
	}); 


	jQuery(window).on('load', function () {
			
		equalheight(".equal-wraper .equal-col"),
		
		animate_content(),
				
		masonryBox(),
	
		page_loader(),

		work_carousel()		 
});



	jQuery(window).on('scroll', function () {

		color_fill_header()
	});
	


	jQuery(window).on('resize', function () {
	 
	 	footer_fixed()
	});


	jQuery(document).on('submit', 'form.cons-contact-form', function(e){
		e.preventDefault();

		jQuery.ajax({
			url: 'http://thememajestic.com/modern/phpmailer/mail.php',
			data: form.serialize() + "&action=contactform",
			type: 'POST',
			dataType: 'JSON',
			beforeSend: function() {
				jQuery('.loading-area').show();
			},

			success:function(data){
				jQuery('.loading-area').hide();
				if(data['success']){
				jQuery("<div class='alert alert-success'>"+data['message']+"</div>").insertBefore('form.cons-contact-form');
				}else{
				jQuery("<div class='alert alert-danger'>"+data['message']+"</div>").insertBefore('form.cons-contact-form');	
				}
			}
		});
		return false;
	});	



	jQuery('.vnav-btn').on('click', function () { 
		jQuery('.nav-sidebar').toggleClass('active');
		
	});
	jQuery('.vnav-close').on('click', function () { 
		jQuery('.nav-sidebar').removeClass('active');
		
	});



	    jQuery('.nav-transparent-area').fadeOut(500);	
		jQuery('.vnav-btn').on('click', function () {	
			if (jQuery(this).hasClass('open')) {
				jQuery('.nav-transparent-area').animate({'left': '100%'});
			} 
			else {
				if (jQuery(this).hasClass('closed')) {
				jQuery('.nav-transparent-area').fadeIn(500);
				}
			}
		});
						
		jQuery('.nav-transparent-area').on('click', function () {	
				jQuery('.nav-sidebar').animate({'right': '-500px'});
				jQuery('.nav-transparent-area').fadeOut(500);
				jQuery('.vnav-btn').addClass('closed');
		});
			


})(window.jQuery);