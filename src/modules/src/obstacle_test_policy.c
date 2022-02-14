#include "network_evaluate.h"
#include "debug.h"
#include "param.h"
#include <string.h>

static const int K_OBSTACLES = 2; // k-nearest obstacles
static const int NEIGHBORS = 3;
static const int OBST_OBS_DIM = 4;
static const int NBR_OBS_DIM = 6;

float linear(float num) {
	return num;
}

typedef struct control_t_n {
	float thrust_0;
	float thrust_1;
	float thrust_2;
	float thrust_3;
} control_t_n;

void networkEvaluate(control_t_n* control_n, const float* state_array);
void neighborEmbeddings(const float neighbor_array[NEIGHBORS][NBR_OBS_DIM]);
void obstacleEmbeddings(const float obstacles_array[K_OBSTACLES][OBST_OBS_DIM]);


float sigmoid(float num) {
	return 1 / (1 + exp(-num));
}



float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

//static const int structure [8][2] = {{6, 4},{4, 4},{24, 12},{12, 12},{4, 4},{4, 4},{20, 24},{24, 4}};
static const int structure [8][2] = {{24, 12}, {12, 12}, {6, 4}, {4, 4}, {4, 4}, {4, 4}, {20, 24}, {24, 4}};
static float output_0[12];
static float output_1[12];
static float output_2[4];
static float output_3[4];
static float output_4[4];
static float output_5[4];
static float output_6[24];
static float output_7[4];
static float neighbor_embeds[4];
static float obstacle_embeds[4];
static float embedding_array[20];
static const float actor_encoder_neighbor_encoder_embedding_mlp_0_weight[6][4] = {{-0.32565340399742126,0.2056400626897812,-0.20977051556110382,1.5410555601119995},{0.6266855001449585,-0.24734915792942047,-0.3706405758857727,-0.5967013239860535},{0.8015619516372681,-0.0743730291724205,-0.13086684048175812,0.21602658927440643},{-0.23373815417289734,-0.3884088099002838,0.12105350196361542,-0.3817952275276184},{0.4990532696247101,0.1948128193616867,-0.28847190737724304,0.16685698926448822},{0.6095005869865417,-0.08113326877355576,-0.2855131924152374,0.13133005797863007}};
static const float actor_encoder_neighbor_encoder_embedding_mlp_2_weight[4][4] = {{0.9856160879135132,-0.5977442264556885,-0.04831280559301376,-0.870262861251831},{-0.2477561980485916,0.2492833435535431,0.035218626260757446,-0.6790463924407959},{-0.27243199944496155,0.07037417590618134,0.7020437717437744,0.5567783713340759},{-1.0813404321670532,0.3617035150527954,0.18963941931724548,-0.7243682146072388}};
static const float actor_encoder_self_encoder_0_weight[24][12] = {{0.07866977900266647,0.021208468824625015,-0.02792726270854473,0.049796391278505325,0.04555093124508858,-0.004308253992348909,0.03406040370464325,-0.1162470132112503,0.014562233351171017,0.07967580854892731,-0.07763177156448364,-0.037946365773677826},{0.15324369072914124,0.022813517600297928,0.3611333966255188,-0.02747401036322117,-0.012906915508210659,-0.13989463448524475,-0.10123945772647858,0.0023857755586504936,-0.05727604404091835,-0.14330780506134033,-0.024013463407754898,0.07832635194063187},{-0.09108632057905197,-0.10482332110404968,-0.12606972455978394,0.17682407796382904,0.20485278964042664,-0.2039206326007843,0.3023393750190735,-0.15734843909740448,-0.13367308676242828,-0.3593425452709198,-0.24642446637153625,-0.081443190574646},{0.01968698762357235,0.084224171936512,0.3690681755542755,0.35842427611351013,-0.038552116602659225,-0.004065340384840965,-0.21834279596805573,-0.6005088686943054,0.22030673921108246,0.3074718713760376,0.003847392275929451,-0.014139282517135143},{0.28526952862739563,0.08628258854150772,0.19533707201480865,0.025907790288329124,0.22454361617565155,0.3099534809589386,-0.030396267771720886,0.006435253191739321,-0.415768027305603,-0.10948790609836578,0.03303387016057968,0.33371293544769287},{-0.3179510831832886,-0.1816212385892868,-0.3982671797275543,0.042082466185092926,0.2921692728996277,0.04884317144751549,0.3834897577762604,-0.034614309668540955,-0.04492078721523285,-0.44701364636421204,0.1444842666387558,0.02331598475575447},{0.8241313695907593,-0.18569064140319824,-0.5565136075019836,0.8723165988922119,-0.4313502609729767,-0.24419641494750977,-0.5278490781784058,0.263497531414032,-0.32141396403312683,0.37538859248161316,0.31012946367263794,-0.2553759515285492},{0.2524702250957489,-0.10176463425159454,0.09294681996107101,-0.4877825081348419,-0.10924050211906433,0.07849416881799698,0.13578355312347412,-0.01897423528134823,0.3523557186126709,0.03374101594090462,0.060833610594272614,-0.21490639448165894},{0.27348563075065613,-0.6473070979118347,0.22480039298534393,0.597037672996521,0.0555025152862072,0.006396756507456303,-0.3153984248638153,-1.2287561893463135,0.41548240184783936,0.618617832660675,-0.12002582848072052,-0.2674344480037689},{-0.22076667845249176,0.42760711908340454,0.09621287137269974,0.34282833337783813,-0.0036525132600218058,-0.35597026348114014,-0.11037064343690872,0.05700400471687317,0.28471872210502625,0.11266519129276276,0.6216466426849365,-0.2819613218307495},{0.6973546743392944,-0.629030168056488,-0.4058670103549957,0.4586978256702423,-0.373173326253891,0.033616725355386734,-0.08239268511533737,-0.10409994423389435,-0.20753085613250732,-0.24050958454608917,0.5655202269554138,-0.5391328930854797},{0.4639091193675995,-0.30913347005844116,0.4190214276313782,-0.11140724271535873,0.028695175424218178,-0.32298365235328674,-0.28224417567253113,-0.13936491310596466,-1.1408891677856445,0.024299122393131256,0.2918435335159302,1.184969186782837},{-0.16201412677764893,-0.03997435420751572,-0.12007637321949005,-0.4528059661388397,-0.5659644603729248,-0.07031265646219254,0.6586250066757202,-0.1886514127254486,0.9201148748397827,0.4038599729537964,0.009386979043483734,-1.0061265230178833},{0.20483063161373138,-0.25515109300613403,-0.04218294844031334,-0.03167291730642319,-0.19363506138324738,-0.5896700024604797,-0.5181480050086975,-1.2122315168380737,0.008742408826947212,0.10674212127923965,0.37456250190734863,0.05828656256198883},{-0.26543447375297546,0.047296881675720215,0.23406882584095,0.5661408305168152,0.3244935870170593,-0.12997806072235107,-0.4366114139556885,-0.21353065967559814,0.05656442418694496,-0.009657510556280613,-0.004979534074664116,-0.0112460907548666},{-0.09006989002227783,-0.20528580248355865,0.2365655153989792,0.3509165644645691,0.21175605058670044,-0.03615111857652664,-0.28202736377716064,-0.22307594120502472,-0.04825741425156593,0.25367358326911926,0.06564299762248993,0.023716509342193604},{-0.03850018233060837,-0.04908289015293121,0.35663875937461853,0.23779816925525665,-0.0735984668135643,0.19998012483119965,0.23647065460681915,0.070360466837883,-0.21259140968322754,-0.3719808757305145,0.0371326245367527,0.2926541268825531},{-0.33940809965133667,0.4185166358947754,0.060193680226802826,-0.44190481305122375,0.292216032743454,0.02370750531554222,0.23237475752830505,0.01676042005419731,0.13977432250976562,0.15145361423492432,-0.23298291862010956,0.20927540957927704},{0.01326047908514738,-0.3281300663948059,0.41781681776046753,0.24282728135585785,-0.05321703478693962,0.12093939632177353,-0.1687166392803192,-0.03188153728842735,0.25953027606010437,0.34309932589530945,0.009533940814435482,0.049223825335502625},{-0.14561401307582855,0.23526631295681,0.39714518189430237,0.42802947759628296,-0.24163281917572021,0.02354508265852928,0.09577677398920059,-0.07313103973865509,-0.3586271405220032,0.16071200370788574,0.3676300346851349,0.09176658093929291},{-0.0336606502532959,-0.13393425941467285,0.0915752649307251,-0.2905225455760956,0.27493780851364136,-0.4465470314025879,0.19463008642196655,-0.07336065918207169,-0.30120986700057983,0.06541833281517029,0.2864762544631958,-0.2660401463508606},{0.21826724708080292,0.1016961857676506,-0.1491660177707672,0.0499616339802742,-0.3803514242172241,0.23118381202220917,0.42673259973526,-0.0035463632084429264,-0.33955085277557373,-0.37538856267929077,-0.2616705000400543,-0.05618267506361008},{-0.4288649260997772,0.14130020141601562,0.08276396244764328,0.2234858125448227,-0.3070225715637207,-0.2196129858493805,-0.17886926233768463,-0.044160231947898865,0.36164066195487976,-0.26705893874168396,-0.10176094621419907,0.10255824774503708},{-0.17642799019813538,0.06752004474401474,-0.00996793620288372,-0.20473982393741608,-0.2795071303844452,-0.08020443469285965,-0.07862222194671631,-0.09592103958129883,0.003395671723410487,0.12941314280033112,-0.09829354286193848,-0.0992884412407875}};
static const float actor_encoder_self_encoder_2_weight[12][12] = {{-0.6232559084892273,0.11268610507249832,-0.024560395628213882,-0.023643895983695984,0.3233986496925354,0.33678457140922546,0.005201386287808418,0.2609524130821228,0.32477614283561707,-0.32834842801094055,0.04219986870884895,-0.015159924514591694},{-0.05496646836400032,-0.0836806520819664,-0.1202632412314415,-0.35065957903862,0.0225849449634552,0.3858545124530792,-0.4387977421283722,-0.18140879273414612,-0.43756455183029175,0.060784805566072464,-0.2605556547641754,0.17815981805324554},{0.5499392747879028,0.34822148084640503,-0.2793440818786621,-0.1980462670326233,0.22565586864948273,-0.1275559514760971,-0.07366872578859329,-0.04953227564692497,-0.07677692919969559,0.30498912930488586,0.047029413282871246,-0.05218689143657684},{0.3420256972312927,0.121998131275177,0.3269147574901581,-0.19679895043373108,0.14906327426433563,0.14713254570960999,-0.04585836082696915,0.1585606038570404,-0.040431562811136246,-0.43793177604675293,0.2777717113494873,-0.06085239723324776},{0.16860392689704895,0.4852897822856903,-0.24057850241661072,0.4273819625377655,-0.2736814320087433,-0.11485615372657776,-0.4031006693840027,0.1212855651974678,-0.032532576471567154,0.17855137586593628,-0.028019234538078308,-0.2898254692554474},{-0.09686758369207382,-0.054288823157548904,0.11068034172058105,0.08553336560726166,0.4852845370769501,0.24442698061466217,0.17327509820461273,0.22246398031711578,-0.14151272177696228,-0.05195293948054314,-0.054231658577919006,0.43554380536079407},{-0.1075429692864418,0.3639954626560211,-0.5053998231887817,-0.3939603269100189,0.19795912504196167,-0.3359996974468231,0.22779645025730133,-0.13577216863632202,-0.23816312849521637,0.1759568154811859,-0.11469709128141403,0.12257032841444016},{0.09641096740961075,-0.5238824486732483,0.24286696314811707,-0.4935779571533203,0.23266683518886566,-0.4269137382507324,-0.29467353224754333,-0.3188965320587158,-0.18512794375419617,-0.3926800489425659,-0.46037742495536804,0.08552901446819305},{0.24511122703552246,0.3740066587924957,0.3237461447715759,-0.26792752742767334,-0.41943252086639404,0.2851428985595703,0.6186988353729248,0.0886625275015831,-0.18691393733024597,-0.06343494355678558,-0.053710196167230606,0.26142656803131104},{0.47566086053848267,0.037157390266656876,-0.28325778245925903,-0.13340240716934204,0.038546618074178696,0.21993203461170197,0.31808510422706604,0.20945638418197632,0.183503195643425,-0.20132018625736237,0.2644638419151306,0.4760771095752716},{-0.052104439586400986,-0.024337276816368103,-0.34278011322021484,-0.4807564318180084,-0.4029691517353058,0.3614978492259979,-0.22842276096343994,0.2803869843482971,0.25108253955841064,-0.4034981429576874,0.3197716772556305,-0.3132101893424988},{0.14375123381614685,-0.19546133279800415,-0.4402516484260559,0.2330726683139801,-0.04338524490594864,-0.3031524121761322,-0.4491881728172302,0.38487470149993896,0.1086118072271347,0.03586742654442787,0.11418768763542175,0.3396346867084503}};
static const float actor_encoder_obstacle_encoder_0_weight[4][4] = {{-0.40028131008148193,1.6833323240280151,0.007705762051045895,0.5572752356529236},{-0.142246276140213,0.07478010654449463,-2.4569449424743652,0.6104051470756531},{0.0016471963608637452,-0.17430543899536133,-0.07481326907873154,3.747591472347267e-05},{-0.49458327889442444,0.8215885162353516,-0.36618921160697937,0.4676210284233093}};
static const float actor_encoder_obstacle_encoder_2_weight[4][4] = {{0.7796843647956848,0.5815191864967346,0.4561319947242737,-0.6628708839416504},{0.13985329866409302,0.5037147998809814,-0.8591303825378418,-0.23640881478786469},{0.506564199924469,-0.2599029242992401,0.9310889840126038,0.1165623739361763},{0.08788999170064926,0.19133344292640686,-0.7960367798805237,0.6432652473449707}};
static const float actor_encoder_feed_forward_0_weight[20][24] = {{-0.13921953737735748,0.16280284523963928,-0.26330915093421936,0.18458859622478485,-0.5383927822113037,-0.12229888886213303,0.2221318930387497,-0.43388745188713074,-0.16799142956733704,-0.3824062943458557,-0.4208132028579712,-0.43749260902404785,0.09198827296495438,-0.22511298954486847,0.11187812685966492,0.05966481193900108,0.040675535798072815,-0.09406648576259613,-0.15952622890472412,-0.11112338304519653,-0.15532667934894562,-0.335568368434906,0.01008972991257906,-0.15702112019062042},{0.22831784188747406,-0.02669093944132328,-0.02014041692018509,0.022497287020087242,0.040563564747571945,0.040449436753988266,0.29406100511550903,0.01770738512277603,-0.2514657974243164,-0.20894154906272888,0.05100146308541298,-0.04583175480365753,0.13673478364944458,0.1677166223526001,0.15180464088916779,0.2736668288707733,-0.1677119880914688,0.1935102939605713,0.2834150791168213,-0.05239073932170868,0.3112308979034424,0.38443273305892944,0.21367093920707703,-0.16080154478549957},{0.23091425001621246,-0.2875518500804901,-0.27149495482444763,0.34033849835395813,0.3828941285610199,0.07713326066732407,0.18536309897899628,-0.2094450443983078,-0.1459205150604248,0.0053330897353589535,0.4024348855018616,-0.13329945504665375,-0.26124563813209534,0.011634614318609238,0.07210379838943481,-0.07168783247470856,0.03428692743182182,-0.30590757727622986,0.2817839980125427,0.02365976758301258,0.2884647846221924,-0.07583268731832504,-0.12831328809261322,-0.04316702112555504},{0.28776484727859497,-0.11182688176631927,0.0461769737303257,0.3239147961139679,0.0577850379049778,-0.018920617178082466,-0.1046919971704483,0.4552658498287201,0.008307890966534615,-0.3560914099216461,0.025238312780857086,-0.19889312982559204,0.11407612264156342,0.02119411714375019,-0.29851359128952026,0.06823546439409256,0.0019872146658599377,-0.2500910758972168,-0.2116268277168274,-0.38059964776039124,0.2575962245464325,-0.29119208455085754,-0.07898561656475067,-0.2224445790052414},{-0.08416339755058289,-0.2733898460865021,0.03089108131825924,0.1361987590789795,0.22799170017242432,0.09910591691732407,0.1851644515991211,0.06456228345632553,-0.12151020020246506,-0.0692654550075531,-0.1532779335975647,-0.08888588100671768,0.2369951754808426,0.11994585394859314,0.274059921503067,0.11440441012382507,-0.011198125779628754,0.22459398210048676,0.09426969289779663,0.0070419092662632465,0.01531513687223196,-0.13413746654987335,-0.023189647123217583,-0.268709659576416},{0.16575297713279724,-0.19560502469539642,-0.2666980028152466,-0.16322281956672668,0.2921496331691742,0.046832792460918427,0.04923763498663902,-0.12530484795570374,-0.2717605531215668,0.052147336304187775,0.1683020144701004,-0.32455480098724365,-0.2531207203865051,-0.21271072328090668,-0.08204542100429535,0.009376419708132744,0.1288127601146698,-0.032726701349020004,-0.33650487661361694,-0.050698988139629364,-0.2745586335659027,-0.2616277039051056,-0.12219368666410446,0.2556043863296509},{-0.051052287220954895,0.012323142029345036,0.2653522193431854,-0.1730164885520935,0.2694074511528015,0.27348679304122925,0.036149200052022934,-0.1678328514099121,0.24699050188064575,-0.33575257658958435,0.19744347035884857,0.12356960773468018,0.20939739048480988,-0.2566084563732147,0.09447437524795532,0.2612982392311096,0.2822556793689728,-0.14275650680065155,0.08338016271591187,-0.2825305461883545,-0.18076933920383453,0.023699160665273666,-0.20852933824062347,-0.1368720680475235},{-0.11013193428516388,0.28129062056541443,0.1636722981929779,0.3985627293586731,0.24245058000087738,0.014048709534108639,0.38207340240478516,0.16596488654613495,-0.164616197347641,-0.12920647859573364,0.17265847325325012,0.34486645460128784,-0.07207906991243362,-0.2947167456150055,0.24623528122901917,0.05466877296566963,0.07725179940462112,0.25149255990982056,-0.21993844211101532,0.3661918342113495,-0.2771041691303253,0.22847101092338562,0.025337256491184235,0.025159310549497604},{-0.09202097356319427,-0.016209231689572334,-0.28012940287590027,-0.0881410539150238,-0.0389912873506546,0.06478415429592133,-0.2496463507413864,-0.2508092224597931,0.024412458762526512,0.18428727984428406,0.0963379368185997,-0.06343058496713638,0.034806475043296814,0.11947961151599884,0.36577168107032776,0.09627045691013336,0.13182832300662994,-0.24647106230258942,-0.047075070440769196,-0.1346418410539627,0.16629430651664734,-0.14372019469738007,0.28252077102661133,-0.20840498805046082},{-0.07796677201986313,-0.044777173548936844,0.33434396982192993,0.10204511880874634,0.38623008131980896,0.016745034605264664,0.049575500190258026,-0.15830232203006744,-0.23056088387966156,0.06999196112155914,0.0402526818215847,-0.3337417542934418,-0.19551479816436768,-0.2739781439304352,-0.35258749127388,-0.10515670478343964,-0.4353843033313751,-0.2466573268175125,0.2825363576412201,-0.376838356256485,0.3473547399044037,-0.08369730412960052,0.20577819645404816,0.021848903968930244},{0.14958369731903076,0.15432384610176086,-0.5445035099983215,0.2169264256954193,0.10363487154245377,-0.14008699357509613,-0.030397998169064522,-0.19864894449710846,-0.3041006326675415,0.18158142268657684,0.1799290031194687,-0.20272627472877502,-0.16213564574718475,-0.04675743728876114,-0.2926258444786072,0.1826935112476349,0.12117790430784225,-0.24777758121490479,-0.0245965588837862,0.20326140522956848,-0.30122166872024536,0.3351582884788513,-0.3166636526584625,0.008753144182264805},{-0.3306121230125427,-0.4335527718067169,-0.01769298128783703,0.30728670954704285,0.016393113881349564,-0.5015280842781067,-0.3106532692909241,0.13280092179775238,0.17557799816131592,0.13686564564704895,-0.1665014624595642,-0.08441436290740967,-0.39166930317878723,-0.36125609278678894,0.0021582229528576136,-0.0031175408512353897,0.0651717409491539,0.045736268162727356,-0.1619153618812561,-0.08456676453351974,-0.1917370706796646,-0.2757093906402588,0.23994037508964539,0.02315639518201351},{-0.38264426589012146,0.24525152146816254,0.1287783831357956,0.14399217069149017,0.21480517089366913,-0.2901539206504822,-0.15364766120910645,-0.21293918788433075,0.1230146735906601,0.263899028301239,0.02380225621163845,-0.25361737608909607,0.2957318425178528,0.24402739107608795,0.036308303475379944,-0.1345255970954895,0.10736814141273499,-0.043234821408987045,-0.24518871307373047,-0.1400909274816513,0.19401484727859497,-0.23447377979755402,-0.21832458674907684,-0.016856910660862923},{-0.345011442899704,-0.07757925242185593,0.21859222650527954,0.1768539696931839,-0.3656781315803528,-0.2522282898426056,-0.036115773022174835,-0.14422793686389923,0.01516714133322239,0.02754070982336998,0.051403727382421494,-0.07794274389743805,-0.09515971690416336,0.2039683759212494,0.12959538400173187,-0.28827956318855286,-0.125787615776062,0.2634185552597046,-0.3272680342197418,0.3409171998500824,0.1682262122631073,-0.2941778600215912,0.22480449080467224,0.0800834596157074},{0.12190642952919006,-0.03257371112704277,-0.2174726128578186,-0.046942707151174545,0.3728038966655731,0.43560829758644104,-0.16708162426948547,0.06094692647457123,0.3279472589492798,-0.16188623011112213,-0.1075703352689743,-0.3106943666934967,-0.006108706817030907,0.24766793847084045,0.2978091537952423,-0.24766413867473602,0.18873381614685059,-0.08571892976760864,0.06112213432788849,-0.37430140376091003,0.14760224521160126,0.38554808497428894,0.2414315938949585,-0.050655387341976166},{-0.2641682028770447,-0.12164375931024551,-0.08196725696325302,0.203170046210289,-0.4056587517261505,-0.1884537637233734,0.06605411320924759,-0.070279560983181,-0.17689546942710876,-0.32310599088668823,-0.00851765088737011,-0.3299662470817566,-0.006283948663622141,-0.09703389555215836,0.3422350585460663,0.23417750000953674,0.019310252740979195,0.23105381429195404,0.1596156805753708,0.02482134848833084,0.1785983443260193,-0.026901016011834145,-0.10069823265075684,0.24078328907489777},{0.2503735423088074,0.01013406366109848,0.15147827565670013,-0.10520081222057343,-0.33511173725128174,-0.16949471831321716,-0.2515505850315094,0.0819617435336113,0.2709805369377136,-0.4534685015678406,0.15932618081569672,0.15508154034614563,0.17821897566318512,0.04399567097425461,0.2808155417442322,-0.07073383033275604,-0.11211559176445007,-0.15538962185382843,0.23421445488929749,0.21264244616031647,0.020033782348036766,0.21029944717884064,-0.1194457933306694,-0.46369439363479614},{0.29712575674057007,0.18340979516506195,-0.17644143104553223,0.26425686478614807,-0.0031358127016574144,-0.17848087847232819,0.2819312512874603,-0.24809423089027405,0.513117790222168,-0.316552996635437,-0.22907297313213348,0.31224721670150757,0.0007127542630769312,0.3215845227241516,-0.24479536712169647,-0.03330589085817337,0.027562318369746208,-0.3304515779018402,0.005803997628390789,0.07082800567150116,-0.23817698657512665,0.29239732027053833,0.04653758183121681,-0.022113658487796783},{0.07868035137653351,-0.28613972663879395,0.23364372551441193,-0.26736560463905334,0.07850173115730286,-0.18698154389858246,0.2929561734199524,-0.4285288453102112,-0.028452668339014053,-0.1434844732284546,-0.28492698073387146,0.15247145295143127,-0.27753931283950806,0.4801444411277771,-0.2227209061384201,0.4703696370124817,-0.06298848241567612,-0.30030104517936707,-0.11191356927156448,0.03956880047917366,0.25157469511032104,-0.4494560956954956,-0.11774519830942154,0.3769798278808594},{0.3106600344181061,-0.2088155448436737,0.2774716317653656,0.09076538681983948,0.1493905931711197,0.303063303232193,0.5135444402694702,0.08499319106340408,0.3714338541030884,-0.12437091022729874,-0.5254507660865784,-0.2649429738521576,-0.21551133692264557,0.15473754703998566,0.2610034942626953,0.2584488093852997,0.4220479130744934,-0.38225501775741577,0.3889538049697876,0.28261443972587585,0.08200066536664963,0.14508983492851257,-0.39359065890312195,-0.052969831973314285}};
static const float action_parameterization_distribution_linear_weight[24][4] = {{-0.21847651898860931,0.2814404368400574,0.043719805777072906,-0.2048388570547104},{-0.03683300316333771,-0.09403016418218613,-0.34637054800987244,-0.27982455492019653},{0.05279288813471794,-0.16430506110191345,0.3425930142402649,-0.41011881828308105},{0.3922450840473175,0.3053172528743744,0.06704974174499512,0.14205479621887207},{-0.33255529403686523,-0.08508586138486862,-0.38069218397140503,-0.055424172431230545},{-0.30645015835762024,-0.18636083602905273,-0.15770334005355835,0.18012213706970215},{-0.2949819564819336,0.04462546855211258,0.3805490732192993,-0.06623595207929611},{0.010056748986244202,-0.2410951405763626,-0.3789035677909851,-0.09321386367082596},{-0.31269705295562744,0.0022103209048509598,0.36437809467315674,-0.19441883265972137},{-0.08273003995418549,-0.34306395053863525,0.009128520265221596,0.34704360365867615},{0.006983312778174877,0.3253779411315918,-0.5171082615852356,-0.04222547635436058},{-0.39487674832344055,0.263755202293396,-0.11871226131916046,0.27478593587875366},{0.10928379744291306,-0.25198084115982056,-0.2791961431503296,0.0719866082072258},{-0.49826881289482117,-0.0014896525535732508,-0.24226777255535126,0.1107458546757698},{-0.06769100576639175,0.16704167425632477,0.11908535659313202,0.2184043526649475},{-0.16670607030391693,0.26349011063575745,0.31369808316230774,-0.2743902802467346},{-0.07067758589982986,0.37411364912986755,0.15524810552597046,0.44855430722236633},{0.44434016942977905,-0.2555949091911316,-0.21702256798744202,0.24357913434505463},{-0.15528425574302673,-0.20437830686569214,0.16548778116703033,-0.3805513381958008},{0.01533877570182085,0.19963571429252625,0.17145894467830658,0.40987154841423035},{0.01905728504061699,-0.3213922083377838,0.032525841146707535,0.30146005749702454},{-0.04143579304218292,0.01985754258930683,-0.3012031018733978,-0.3625771701335907},{0.14994864165782928,-0.12566883862018585,-0.18260373175144196,-0.1660202145576477},{-0.3624879717826843,-0.18871749937534332,0.4478333592414856,-0.032401204109191895}};
static const float actor_encoder_neighbor_encoder_embedding_mlp_0_bias[4] = {0.09330518543720245,-0.10550492256879807,0.15016357600688934,-0.14674517512321472};
static const float actor_encoder_neighbor_encoder_embedding_mlp_2_bias[4] = {-0.07397496700286865,-0.14949551224708557,0.026734350249171257,0.07042759656906128};
static const float actor_encoder_self_encoder_0_bias[12] = {-0.01689906232059002,0.14751403033733368,0.1613190770149231,0.07544542849063873,0.042821161448955536,-0.001981601119041443,0.09624975174665451,0.052490636706352234,-0.06530040502548218,-0.05005799978971481,0.038991779088974,0.1667608767747879};
static const float actor_encoder_self_encoder_2_bias[12] = {-0.0562971867620945,0.049667708575725555,-0.007165111135691404,0.027283119037747383,0.006127598229795694,-0.16193513572216034,-0.09292793273925781,0.07489406317472458,-0.011232739314436913,0.10567823052406311,-0.006807616911828518,-0.09899899363517761};
static const float actor_encoder_obstacle_encoder_0_bias[4] = {0.08854065835475922,0.03579501062631607,-0.17514410614967346,-0.2608671486377716};
static const float actor_encoder_obstacle_encoder_2_bias[4] = {0.1748422086238861,-0.05384168401360512,0.0309311430901289,-0.09272832423448563};
static const float actor_encoder_feed_forward_0_bias[24] = {-0.04585983231663704,-0.01423086877912283,-0.006472937297075987,0.009612023830413818,-0.011800412088632584,-0.005475950427353382,-0.05313071236014366,0.048156216740608215,-0.012969196774065495,-0.014534888789057732,-0.008106743916869164,-0.07504358142614365,0.019480057060718536,-0.05892554670572281,0.029741942882537842,-0.0019358359277248383,-0.062370315194129944,-0.03126116469502449,0.06807046383619308,-0.06273075938224792,0.023958837613463402,0.09422431141138077,0.0617804117500782,0.006616531405597925};
static const float action_parameterization_distribution_linear_bias[4] = {-0.0067688580602407455,-0.07106431573629379,0.02324328012764454,-0.029947055503726006};

void neighborEmbeddings(const float neighbors_array[NEIGHBORS][NBR_OBS_DIM]) {
  //reset embeddings accumulator to zero
  memset(neighbor_embeds, 0, sizeof(neighbor_embeds));

  // get the neighbor embeddings
  for (int n = 0; n < NEIGHBORS; n++) {
    for (int i = 0; i < structure[2][1]; i++) {
      output_2[i] = 0;
      for (int j = 0; j < structure[2][0]; j++) {
        output_2[i] += neighbors_array[n][j] * actor_encoder_neighbor_encoder_embedding_mlp_0_weight[j][i];
      }
      output_2[i] += actor_encoder_neighbor_encoder_embedding_mlp_0_bias[i];
      output_2[i] = tanhf(output_2[i]);
    }

    for (int i = 0; i < structure[3][1]; i++) {
      output_3[i] = 0;
      for (int j = 0; j < structure[3][0]; j++) {
        output_3[i] += output_2[j] * actor_encoder_neighbor_encoder_embedding_mlp_2_weight[j][i];
      }

      output_3[i] += actor_encoder_neighbor_encoder_embedding_mlp_2_bias[i];
      output_3[i] = tanhf(output_3[i]);
      neighbor_embeds[i] += output_3[i];
    }
  }

  int self_size = structure[1][1]; // size of self embeddings. Need to offset by this much in the embedding array
  // get mean embeddings
  for (int i = 0; i < structure[3][1]; i++) {
    neighbor_embeds[i] /= NEIGHBORS;
    embedding_array[i+self_size] = neighbor_embeds[i];
  }
}

void obstacleEmbeddings(const float obstacles_array[K_OBSTACLES][OBST_OBS_DIM]) {
    // reset the embeddings accumulator to zero
    memset(obstacle_embeds, 0, sizeof(obstacle_embeds));

    // get the obstacle embeddings
    for (int o = 0; o < K_OBSTACLES; o++) {
        for (int i = 0; i < structure[4][1]; i++) {
            output_4[i] = 0;
            for (int j = 0; j < structure[4][0]; j++) {
                output_4[i] += obstacles_array[o][j] * actor_encoder_obstacle_encoder_0_weight[j][i];
            }
            output_4[i] += actor_encoder_obstacle_encoder_0_bias[i];
            output_4[i] = tanhf(output_4[i]);
        }

        for (int i = 0; i < structure[5][1]; i++) {
            output_5[i] = 0;
            for (int j = 0; j < structure[5][0]; j++) {
                output_5[i] += output_4[j] * actor_encoder_obstacle_encoder_2_weight[j][i];
            }

            output_5[i] += actor_encoder_obstacle_encoder_2_bias[i];
            output_5[i] = tanhf(output_5[i]);
            obstacle_embeds[i] += output_5[i];
        }
    }

    int self_offset = structure[1][1];
    int neighbor_offset = structure[3][1];
    int obst_offset = self_offset + neighbor_offset; // size of self embeddings + neighbor embeddings. Need to offset by this much in the final embeddings array
    // get mean embeddings
    for (int i = 0; i < structure[5][1]; i++) {
        obstacle_embeds[i] /= K_OBSTACLES;
        embedding_array[i+obst_offset]  = obstacle_embeds[i];
    }
}

void networkEval(struct control_t_n *control_n, const float *state_array) {
        for (int i = 0; i < structure[0][1]; i++) {
            output_0[i] = 0;
            for (int j = 0; j < structure[0][0]; j++) {
                output_0[i] += state_array[j] * actor_encoder_self_encoder_0_weight[j][i];
            }
            output_0[i] += actor_encoder_self_encoder_0_bias[i];
            output_0[i] = tanhf(output_0[i]);
        }

        for (int i = 0; i < structure[1][1]; i++) {
            embedding_array[i] = 0;
            for (int j = 0; j < structure[1][0]; j++) {
                embedding_array[i] += output_0[j] * actor_encoder_self_encoder_2_weight[j][i];
            }
            embedding_array[i] += actor_encoder_self_encoder_2_bias[i];
            embedding_array[i] = tanhf(embedding_array[i]);
        }

        // last feedforward and action parameterization layers
        for (int i = 0; i < structure[6][1]; i++) {
            output_6[i] = 0;
            for (int j = 0; j < structure[6][0]; j++) {
                output_6[i] += embedding_array[j] * actor_encoder_feed_forward_0_weight[j][i];
            }
            output_6[i] += actor_encoder_feed_forward_0_bias[i];
            output_6[i] = tanhf(output_6[i]);
        }


        for (int i = 0; i < structure[7][1]; i++) {
            output_7[i] = 0;
            for (int j = 0; j < structure[7][0]; j++) {
                output_7[i] += output_6[j] * action_parameterization_distribution_linear_weight[j][i];
            }
            output_7[i] += action_parameterization_distribution_linear_bias[i];
        }


        control_n->thrust_0 = output_7[0];
        control_n->thrust_1 = output_7[1];
        control_n->thrust_2 = output_7[2];
        control_n->thrust_3 = output_7[3];
    }

