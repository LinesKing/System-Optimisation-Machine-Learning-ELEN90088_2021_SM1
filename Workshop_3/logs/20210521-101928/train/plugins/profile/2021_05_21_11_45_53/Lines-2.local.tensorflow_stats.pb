"??
BHostIDLE"IDLE1     ??@A     ??@a???S???i???S????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     l?@9     l?@A     l?@I     l?@a֥??L???i?>?胊???Unknown?
iHostWriteSummary"WriteSummary(1      \@9      \@A      \@I      \@a0 ?0|?i?>.??????Unknown?
?HostMatMul".gradient_tape/sequential_93/dense_289/MatMul_1(1      ;@9      ;@A      ;@I      ;@a??v?J.[?i>??{????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      :@I      :@a????,Z?i'??[?????Unknown
?HostSquare"2sequential_93/dense_288/ActivityRegularizer/Square(1      8@9      8@A      8@I      8@a???&)X?i?^???????Unknown
lHostIteratorGetNext"IteratorGetNext(1      4@9      4@A      4@I      4@a???fJ"T?ib???????Unknown
vHost_FusedMatMul"sequential_93/dense_289/Relu(1      3@9      3@A      3@I      3@a??-{? S?iGQ?]G????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      0@9      0@A      0@I      0@a@Y??nP?it??U???Unknown
?
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      0@I      0@a@Y??nP?i??C?b???Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a@Y??nP?i?,??p???Unknown
?HostSign"Bgradient_tape/sequential_93/dense_288/ActivityRegularizer/Abs/Sign(1      ,@9      ,@A      ,@I      ,@a0 ?0L?i???|???Unknown
?HostStridedSlice"9sequential_93/dense_288/ActivityRegularizer/strided_slice(1      ,@9      ,@A      ,@I      ,@a0 ?0L?iܬ???#???Unknown
vHost_FusedMatMul"sequential_93/dense_288/Relu(1      ,@9      ,@A      ,@I      ,@a0 ?0L?i?l???*???Unknown
?HostSign"Bgradient_tape/sequential_93/dense_289/ActivityRegularizer/Abs/Sign(1      (@9      (@A      (@I      (@a???&)H?iģwΞ0???Unknown
eHost
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@a???fJ"D?i?Qa?5???Unknown?
yHost_FusedMatMul"sequential_93/dense_290/BiasAdd(1      $@9      $@A      $@I      $@a???fJ"D?i<????:???Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      "@I      "@ah????B?ie?Ϊ7????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a@Y??n@?i?}?>C???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a@Y??n@?i?1+bEG???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a0 ?0<?i??cb?J???Unknown
?HostMatMul",gradient_tape/sequential_93/dense_288/MatMul(1      @9      @A      @I      @a0 ?0<?i???bQN???Unknown
?HostSum"/sequential_93/dense_288/ActivityRegularizer/Sum(1      @9      @A      @I      @a0 ?0<?i?Q?b?Q???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a???&)8?i햇?T???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&)8?i?Y??W???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&)8?i?#??Z???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&)8?ia????]???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a???&)8?i?Z??`???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???&)8?iC?c??c???Unknown
?HostTile">gradient_tape/sequential_93/dense_288/ActivityRegularizer/Tile(1      @9      @A      @I      @a???&)8?i??&d?f???Unknown
?HostTile"@gradient_tape/sequential_93/dense_288/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a???&)8?i%-?? j???Unknown
? HostMatMul",gradient_tape/sequential_93/dense_290/MatMul(1      @9      @A      @I      @a???&)8?i?ȫ?m???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a???fJ"4?it????o???Unknown
?"HostBiasAddGrad"9gradient_tape/sequential_93/dense_288/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???fJ"4?iRvE@r???Unknown
?#HostMatMul",gradient_tape/sequential_93/dense_289/MatMul(1      @9      @A      @I      @a???fJ"4?i0M???t???Unknown
?$HostBiasAddGrad"9gradient_tape/sequential_93/dense_290/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???fJ"4?i$??w???Unknown
?%HostSum"1sequential_93/dense_288/ActivityRegularizer/Sum_1(1      @9      @A      @I      @a???fJ"4?i??+?y???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a@Y??n0?i7??{???Unknown
`'HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a@Y??n0?i????}???Unknown
\(HostGreater"Greater(1      @9      @A      @I      @a@Y??n0?i?1?e????Unknown
V)HostSum"Sum_2(1      @9      @A      @I      @a@Y??n0?iD?Ө????Unknown
?*HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a@Y??n0?icV_A?????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a@Y??n0?i?h6??????Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a@Y??n0?i?z?????Unknown
?-Host
Reciprocal"Igradient_tape/sequential_93/dense_289/ActivityRegularizer/truediv/RealDiv(1      @9      @A      @I      @a@Y??n0?iD?䊶????Unknown
?.HostMatMul".gradient_tape/sequential_93/dense_290/MatMul_1(1      @9      @A      @I      @a@Y??n0?i?????????Unknown
?/HostReadVariableOp"-sequential_93/dense_289/MatMul/ReadVariableOp(1      @9      @A      @I      @a@Y??n0?iڱ?f?????Unknown
?0HostSum"1sequential_93/dense_290/ActivityRegularizer/Sum_1(1      @9      @A      @I      @a@Y??n0?i%?i??????Unknown
?1HostStridedSlice"9sequential_93/dense_290/ActivityRegularizer/strided_slice(1      @9      @A      @I      @a@Y??n0?ip?@Bđ???Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???&)(?i($??F????Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a???&)(?i?qgɔ???Unknown
?4HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a???&)(?i??d?K????Unknown
d5HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a???&)(?iPƋΗ???Unknown
z6HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a???&)(?i['Q????Unknown
|7HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???&)(?i????Ӛ???Unknown
~8HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a???&)(?ix??BV????Unknown
?9HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a???&)(?i0DK?؝???Unknown
?:HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a???&)(?i葬g[????Unknown
?;HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???&)(?i???ݠ???Unknown
?<HostMul"=gradient_tape/sequential_93/dense_288/ActivityRegularizer/Mul(1      @9      @A      @I      @a???&)(?iX-o?`????Unknown
?=HostMul"=gradient_tape/sequential_93/dense_289/ActivityRegularizer/Mul(1      @9      @A      @I      @a???&)(?i{??????Unknown
?>HostTile">gradient_tape/sequential_93/dense_289/ActivityRegularizer/Tile(1      @9      @A      @I      @a???&)(?i??1?e????Unknown
??HostTile"@gradient_tape/sequential_93/dense_289/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a???&)(?i??C?????Unknown
?@HostBiasAddGrad"9gradient_tape/sequential_93/dense_289/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???&)(?i8d??j????Unknown
?AHostTile">gradient_tape/sequential_93/dense_290/ActivityRegularizer/Tile(1      @9      @A      @I      @a???&)(?i??Uh?????Unknown
?BHostTile"@gradient_tape/sequential_93/dense_290/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a???&)(?i????o????Unknown
?CHostAbs"/sequential_93/dense_288/ActivityRegularizer/Abs(1      @9      @A      @I      @a???&)(?i`M??????Unknown
?DHostSquare"2sequential_93/dense_289/ActivityRegularizer/Square(1      @9      @A      @I      @a???&)(?i?yu????Unknown
?EHostSum"/sequential_93/dense_289/ActivityRegularizer/Sum(1      @9      @A      @I      @a???&)(?i??ڱ?????Unknown
?FHostSum"1sequential_93/dense_289/ActivityRegularizer/Sum_1(1      @9      @A      @I      @a???&)(?i?6<Dz????Unknown
?GHostStridedSlice"9sequential_93/dense_289/ActivityRegularizer/strided_slice(1      @9      @A      @I      @a???&)(?i@????????Unknown
?HHostReadVariableOp".sequential_93/dense_289/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???&)(?i???h????Unknown
?IHostAbs"/sequential_93/dense_290/ActivityRegularizer/Abs(1      @9      @A      @I      @a???&)(?i?`?????Unknown
?JHostReadVariableOp".sequential_93/dense_290/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???&)(?ihm???????Unknown
tKHostSigmoid"sequential_93/dense_290/Sigmoid(1      @9      @A      @I      @a???&)(?i ?" ????Unknown
zLHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1       @9       @A       @I       @a@Y??n ?iFD?????Unknown
tMHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a@Y??n ?il???
????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a@Y??n ?i?V?D????Unknown
VOHostCast"Cast(1       @9       @A       @I       @a@Y??n ?i????????Unknown
XPHostCast"Cast_3(1       @9       @A       @I       @a@Y??n ?i?h??????Unknown
XQHostCast"Cast_5(1       @9       @A       @I       @a@Y??n ?i??i????Unknown
XRHostEqual"Equal(1       @9       @A       @I       @a@Y??n ?i*{? ????Unknown
aSHostIdentity"Identity(1       @9       @A       @I       @a@Y??n ?iP?????Unknown?
VTHostMean"Mean(1       @9       @A       @I       @a@Y??n ?iv?j?????Unknown
uUHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a@Y??n ?i?VE????Unknown
fVHostAddN"SGD/gradients/AddN_3(1       @9       @A       @I       @a@Y??n ?iA?????Unknown
jWHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a@Y??n ?i?(-?????Unknown
rXHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a@Y??n ?i?j????Unknown
vYHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a@Y??n ?i4;!????Unknown
vZHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a@Y??n ?iZ??? ????Unknown
v[HostMul"%binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a@Y??n ?i?Mێ"????Unknown
v\HostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a@Y??n ?i???E$????Unknown
v]HostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a@Y??n ?i?_??%????Unknown
?^HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a@Y??n ?i?蝳'????Unknown
}_HostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a@Y??n ?ir?j)????Unknown
``HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a@Y??n ?i>?t!+????Unknown
baHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a@Y??n ?id?`?,????Unknown
wbHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a@Y??n ?i?L?.????Unknown
?cHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a@Y??n ?i??7F0????Unknown
?dHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a@Y??n ?i?#?1????Unknown
?eHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a@Y??n ?i???3????Unknown
?fHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a@Y??n ?i"2?j5????Unknown
?gHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a@Y??n ?iH??!7????Unknown
?hHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a@Y??n ?inD??8????Unknown
?iHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a@Y??n ?i?ͼ?:????Unknown
~jHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a@Y??n ?i?V?F<????Unknown
?kHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a@Y??n ?i?ߓ?=????Unknown
?lHostMul"Egradient_tape/sequential_93/dense_288/ActivityRegularizer/mul_1/Mul_1(1       @9       @A       @I       @a@Y??n ?ii??????Unknown
?mHost
Reciprocal"Igradient_tape/sequential_93/dense_288/ActivityRegularizer/truediv/RealDiv(1       @9       @A       @I       @a@Y??n ?i,?jkA????Unknown
?nHostReluGrad".gradient_tape/sequential_93/dense_288/ReluGrad(1       @9       @A       @I       @a@Y??n ?iR{V"C????Unknown
?oHostMul"?gradient_tape/sequential_93/dense_289/ActivityRegularizer/Mul_1(1       @9       @A       @I       @a@Y??n ?ixB?D????Unknown
?pHostMul"Egradient_tape/sequential_93/dense_289/ActivityRegularizer/mul_1/Mul_1(1       @9       @A       @I       @a@Y??n ?i??-?F????Unknown
?qHostReluGrad".gradient_tape/sequential_93/dense_289/ReluGrad(1       @9       @A       @I       @a@Y??n ?i?GH????Unknown
?rHostSign"Bgradient_tape/sequential_93/dense_290/ActivityRegularizer/Abs/Sign(1       @9       @A       @I       @a@Y??n ?i???I????Unknown
?sHostMul"=gradient_tape/sequential_93/dense_290/ActivityRegularizer/Mul(1       @9       @A       @I       @a@Y??n ?i)??K????Unknown
?tHostMul"Cgradient_tape/sequential_93/dense_290/ActivityRegularizer/mul/Mul_1(1       @9       @A       @I       @a@Y??n ?i6??kM????Unknown
?uHost
Reciprocal"Igradient_tape/sequential_93/dense_290/ActivityRegularizer/truediv/RealDiv(1       @9       @A       @I       @a@Y??n ?i\;?"O????Unknown
?vHostCast"0sequential_93/dense_288/ActivityRegularizer/Cast(1       @9       @A       @I       @a@Y??n ?i?Ĳ?P????Unknown
?wHostAddV2"1sequential_93/dense_288/ActivityRegularizer/add_1(1       @9       @A       @I       @a@Y??n ?i?M??R????Unknown
?xHostMul"/sequential_93/dense_288/ActivityRegularizer/mul(1       @9       @A       @I       @a@Y??n ?i?։GT????Unknown
?yHostRealDiv"3sequential_93/dense_288/ActivityRegularizer/truediv(1       @9       @A       @I       @a@Y??n ?i?_u?U????Unknown
?zHostReadVariableOp".sequential_93/dense_288/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a@Y??n ?i?`?W????Unknown
?{HostAbs"/sequential_93/dense_289/ActivityRegularizer/Abs(1       @9       @A       @I       @a@Y??n ?i@rLlY????Unknown
?|HostCast"0sequential_93/dense_289/ActivityRegularizer/Cast(1       @9       @A       @I       @a@Y??n ?if?7#[????Unknown
?}HostCast"0sequential_93/dense_290/ActivityRegularizer/Cast(1       @9       @A       @I       @a@Y??n ?i??#?\????Unknown
?~HostSquare"2sequential_93/dense_290/ActivityRegularizer/Square(1       @9       @A       @I       @a@Y??n ?i??^????Unknown
?HostSum"/sequential_93/dense_290/ActivityRegularizer/Sum(1       @9       @A       @I       @a@Y??n ?iؖ?G`????Unknown
??HostReadVariableOp"-sequential_93/dense_290/MatMul/ReadVariableOp(1       @9       @A       @I       @a@Y??n ?i???a????Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a@Y??n?i??[??????Unknown
Y?HostCast"Cast_4(1      ??9      ??A      ??I      ??a@Y??n?i$?ѵc????Unknown
U?HostMul"Mul(1      ??9      ??A      ??I      ??a@Y??n?i?mG??????Unknown
}?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a@Y??n?iJ2?le????Unknown
g?HostAddN"SGD/gradients/AddN_1(1      ??9      ??A      ??I      ??a@Y??n?i??2H?????Unknown
g?HostAddN"SGD/gradients/AddN_2(1      ??9      ??A      ??I      ??a@Y??n?ip??#g????Unknown
v?HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a@Y??n?i???????Unknown
x?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a@Y??n?i?D??h????Unknown
z?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a@Y??n?i)	
??????Unknown
??HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a@Y??n?i???j????Unknown
y?HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a@Y??n?iO??l?????Unknown
??Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a@Y??n?i?VkHl????Unknown
??HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a@Y??n?iu?#?????Unknown
??HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a@Y??n?i?V?m????Unknown
??HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a@Y??n?i?????????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a@Y??n?i.iB?o????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a@Y??n?i?-???????Unknown
??HostMul"Agradient_tape/sequential_93/dense_288/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a@Y??n?iT?-mq????Unknown
??HostMul"?gradient_tape/sequential_93/dense_288/ActivityRegularizer/Mul_1(1      ??9      ??A      ??I      ??a@Y??n?i綣H?????Unknown
??HostMul"Cgradient_tape/sequential_93/dense_288/ActivityRegularizer/mul/Mul_1(1      ??9      ??A      ??I      ??a@Y??n?iz{$s????Unknown
??HostMul"Agradient_tape/sequential_93/dense_289/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a@Y??n?i@???????Unknown
??HostMul"Agradient_tape/sequential_93/dense_290/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a@Y??n?i??t????Unknown
??HostMul"?gradient_tape/sequential_93/dense_290/ActivityRegularizer/Mul_1(1      ??9      ??A      ??I      ??a@Y??n?i3?z??????Unknown
??HostMul"Egradient_tape/sequential_93/dense_290/ActivityRegularizer/mul_1/Mul_1(1      ??9      ??A      ??I      ??a@Y??n?iƍ??v????Unknown
??HostSigmoidGrad"9gradient_tape/sequential_93/dense_290/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a@Y??n?iYRfm?????Unknown
??HostMul"1sequential_93/dense_288/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a@Y??n?i??Hx????Unknown
??HostReadVariableOp"-sequential_93/dense_288/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a@Y??n?i?Q$?????Unknown
??HostAddV2"1sequential_93/dense_289/ActivityRegularizer/add_1(1      ??9      ??A      ??I      ??a@Y??n?i???y????Unknown
??HostMul"/sequential_93/dense_289/ActivityRegularizer/mul(1      ??9      ??A      ??I      ??a@Y??n?i?d=??????Unknown
??HostMul"1sequential_93/dense_289/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a@Y??n?i8)??{????Unknown
??HostRealDiv"3sequential_93/dense_289/ActivityRegularizer/truediv(1      ??9      ??A      ??I      ??a@Y??n?i??(??????Unknown
??HostAddV2"1sequential_93/dense_290/ActivityRegularizer/add_1(1      ??9      ??A      ??I      ??a@Y??n?i^??m}????Unknown
??HostMul"/sequential_93/dense_290/ActivityRegularizer/mul(1      ??9      ??A      ??I      ??a@Y??n?i?vI?????Unknown
??HostMul"1sequential_93/dense_290/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a@Y??n?i?;?$????Unknown
??HostRealDiv"3sequential_93/dense_290/ActivityRegularizer/truediv(1      ??9      ??A      ??I      ??a@Y??n?i     ???Unknown
j?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
Z?HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
h?HostMul"Cgradient_tape/sequential_93/dense_289/ActivityRegularizer/mul/Mul_1(i     ???Unknown*??
uHostFlushSummaryWriter"FlushSummaryWriter(1     l?@9     l?@A     l?@I     l?@aA??/?	??iA??/?	???Unknown?
iHostWriteSummary"WriteSummary(1      \@9      \@A      \@I      \@a?r????i^??(????Unknown?
?HostMatMul".gradient_tape/sequential_93/dense_289/MatMul_1(1      ;@9      ;@A      ;@I      ;@a????_ۋ?ie5??rG???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      :@I      :@a????ӊ?il5??????Unknown
?HostSquare"2sequential_93/dense_288/ActivityRegularizer/Square(1      8@9      8@A      8@I      8@a?????irѨ?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      4@9      4@A      4@I      4@aJv????iw?^?Uh???Unknown
vHost_FusedMatMul"sequential_93/dense_289/Relu(1      3@9      3@A      3@I      3@a:???_???i|%!?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      0@9      0@A      0@I      0@a??????i?? ?????Unknown
?	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      0@9      0@A      0@I      0@a??????i???:???Unknown
g
HostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a??????i???|???Unknown
?HostSign"Bgradient_tape/sequential_93/dense_288/ActivityRegularizer/Abs/Sign(1      ,@9      ,@A      ,@I      ,@a?r??|?i??%?????Unknown
?HostStridedSlice"9sequential_93/dense_288/ActivityRegularizer/strided_slice(1      ,@9      ,@A      ,@I      ,@a?r??|?i?k>e????Unknown
vHost_FusedMatMul"sequential_93/dense_288/Relu(1      ,@9      ,@A      ,@I      ,@a?r??|?i?OW,*???Unknown
?HostSign"Bgradient_tape/sequential_93/dense_289/ActivityRegularizer/Abs/Sign(1      (@9      (@A      (@I      (@a?????x?i?7??[???Unknown
eHost
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@aJv??t?i?#??????Unknown?
yHost_FusedMatMul"sequential_93/dense_290/BiasAdd(1      $@9      $@A      $@I      $@aJv??t?i?G<????Unknown
dHostDataset"Iterator::Model(1      9@9      9@A      "@I      "@a)????r?i?}??`????Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a????p?i?m.?d????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a????p?i?]??h???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a?r??l?i??6L2???Unknown
?HostMatMul",gradient_tape/sequential_93/dense_288/MatMul(1      @9      @A      @I      @a?r??l?i?AÖ/O???Unknown
?HostSum"/sequential_93/dense_288/ActivityRegularizer/Sum(1      @9      @A      @I      @a?r??l?i??Ol???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?????h?i???ք???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????h?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????h?i??&\????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????h?i???????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?????h?i?w`?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?????h?i?k?? ???Unknown
?HostTile">gradient_tape/sequential_93/dense_288/ActivityRegularizer/Tile(1      @9      @A      @I      @a?????h?i?_?h???Unknown
?HostTile"@gradient_tape/sequential_93/dense_288/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a?????h?i?S7+2???Unknown
?HostMatMul",gradient_tape/sequential_93/dense_290/MatMul(1      @9      @A      @I      @a?????h?i?G??J???Unknown
? HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aJv??d?i?????_???Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_93/dense_288/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aJv??d?i?3/3t???Unknown
?"HostMatMul",gradient_tape/sequential_93/dense_289/MatMul(1      @9      @A      @I      @aJv??d?i??ܑՈ???Unknown
?#HostBiasAddGrad"9gradient_tape/sequential_93/dense_290/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aJv??d?i??x????Unknown
?$HostSum"1sequential_93/dense_288/ActivityRegularizer/Sum_1(1      @9      @A      @I      @aJv??d?i??7?????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????`?i?????????Unknown
`&HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a????`?i??????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a????`?i?}q??????Unknown
V(HostSum"Sum_2(1      @9      @A      @I      @a????`?i?u/?"????Unknown
?)HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a????`?i?m폤???Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a????`?i?e??&???Unknown
?+HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a????`?i?]i??%???Unknown
?,Host
Reciprocal"Igradient_tape/sequential_93/dense_289/ActivityRegularizer/truediv/RealDiv(1      @9      @A      @I      @a????`?i?U'?*6???Unknown
?-HostMatMul".gradient_tape/sequential_93/dense_290/MatMul_1(1      @9      @A      @I      @a????`?i?M厬F???Unknown
?.HostReadVariableOp"-sequential_93/dense_289/MatMul/ReadVariableOp(1      @9      @A      @I      @a????`?i?E??.W???Unknown
?/HostSum"1sequential_93/dense_290/ActivityRegularizer/Sum_1(1      @9      @A      @I      @a????`?i?=a??g???Unknown
?0HostStridedSlice"9sequential_93/dense_290/ActivityRegularizer/strided_slice(1      @9      @A      @I      @a????`?i?5?2x???Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?????X?iͯ??????Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?????X?i?)???????Unknown
?3HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?????X?iϣ?W????Unknown
d4HostAddN"SGD/gradients/AddN(1      @9      @A      @I      @a?????X?i?Y??????Unknown
z5HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?????X?iї'????Unknown
|6HostSelect"(binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?????X?i???{????Unknown
~7HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a?????X?iӋ??????Unknown
?8HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?????X?i???>????Unknown
?9HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?????X?i?a?????Unknown
?:HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?????X?i??/?????Unknown
?;HostMul"=gradient_tape/sequential_93/dense_288/ActivityRegularizer/Mul(1      @9      @A      @I      @a?????X?i?s?c ???Unknown
?<HostMul"=gradient_tape/sequential_93/dense_289/ActivityRegularizer/Mul(1      @9      @A      @I      @a?????X?i??̋????Unknown
?=HostTile">gradient_tape/sequential_93/dense_289/ActivityRegularizer/Tile(1      @9      @A      @I      @a?????X?i?g?&???Unknown
?>HostTile"@gradient_tape/sequential_93/dense_289/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a?????X?i??i??%???Unknown
??HostBiasAddGrad"9gradient_tape/sequential_93/dense_289/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?????X?i?[8?1???Unknown
?@HostTile">gradient_tape/sequential_93/dense_290/ActivityRegularizer/Tile(1      @9      @A      @I      @a?????X?i???J>???Unknown
?AHostTile"@gradient_tape/sequential_93/dense_290/ActivityRegularizer/Tile_1(1      @9      @A      @I      @a?????X?i?O?
?J???Unknown
?BHostAbs"/sequential_93/dense_288/ActivityRegularizer/Abs(1      @9      @A      @I      @a?????X?i?ɣ?W???Unknown
?CHostSquare"2sequential_93/dense_289/ActivityRegularizer/Square(1      @9      @A      @I      @a?????X?i?Cr
oc???Unknown
?DHostSum"/sequential_93/dense_289/ActivityRegularizer/Sum(1      @9      @A      @I      @a?????X?i??@??o???Unknown
?EHostSum"1sequential_93/dense_289/ActivityRegularizer/Sum_1(1      @9      @A      @I      @a?????X?i?7
2|???Unknown
?FHostStridedSlice"9sequential_93/dense_289/ActivityRegularizer/strided_slice(1      @9      @A      @I      @a?????X?i??݉?????Unknown
?GHostReadVariableOp".sequential_93/dense_289/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?????X?i?+?	?????Unknown
?HHostAbs"/sequential_93/dense_290/ActivityRegularizer/Abs(1      @9      @A      @I      @a?????X?i??z?V????Unknown
?IHostReadVariableOp".sequential_93/dense_290/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?????X?i?I	?????Unknown
tJHostSigmoid"sequential_93/dense_290/Sigmoid(1      @9      @A      @I      @a?????X?i???????Unknown
zKHostAddN"(ArithmeticOptimizer/AddOpsRewrite_AddN_1(1       @9       @A       @I       @a????P?i????Z????Unknown
tLHostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a????P?i??Ո?????Unknown
vMHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a????P?i鍴??????Unknown
VNHostCast"Cast(1       @9       @A       @I       @a????P?iꉓ?????Unknown
XOHostCast"Cast_3(1       @9       @A       @I       @a????P?i??r?^????Unknown
XPHostCast"Cast_5(1       @9       @A       @I       @a????P?i??Q??????Unknown
XQHostEqual"Equal(1       @9       @A       @I       @a????P?i?}0??????Unknown
aRHostIdentity"Identity(1       @9       @A       @I       @a????P?i?y?!????Unknown?
VSHostMean"Mean(1       @9       @A       @I       @a????P?i?u??b???Unknown
uTHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a????P?i?q͇????Unknown
fUHostAddN"SGD/gradients/AddN_3(1       @9       @A       @I       @a????P?i?m??????Unknown
jVHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a????P?i?i??%???Unknown
rWHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a????P?i?ej?f%???Unknown
vXHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a????P?i?aI??-???Unknown
vYHostNeg"%binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a????P?i?](??5???Unknown
vZHostMul"%binary_crossentropy/logistic_loss/mul(1       @9       @A       @I       @a????P?i?Y?)>???Unknown
v[HostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a????P?i?U??jF???Unknown
v\HostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a????P?i?Qņ?N???Unknown
?]HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a????P?i?M???V???Unknown
}^HostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a????P?i?I??-_???Unknown
`_HostDivNoNan"
div_no_nan(1       @9       @A       @I       @a????P?i?Eb?ng???Unknown
b`HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a????P?i?AA??o???Unknown
waHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a????P?i?= ??w???Unknown
?bHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a????P?i?9??1????Unknown
?cHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1       @9       @A       @I       @a????P?i?5ޅr????Unknown
?dHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a????P?i 2???????Unknown
?eHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a????P?i.???????Unknown
?fHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a????P?i*{?5????Unknown
?gHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a????P?i&Z?v????Unknown
?hHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a????P?i"9??????Unknown
~iHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a????P?i??????Unknown
?jHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a????P?i??9????Unknown
?kHostMul"Egradient_tape/sequential_93/dense_288/ActivityRegularizer/mul_1/Mul_1(1       @9       @A       @I       @a????P?iքz????Unknown
?lHost
Reciprocal"Igradient_tape/sequential_93/dense_288/ActivityRegularizer/truediv/RealDiv(1       @9       @A       @I       @a????P?i???????Unknown
?mHostReluGrad".gradient_tape/sequential_93/dense_288/ReluGrad(1       @9       @A       @I       @a????P?i	???????Unknown
?nHostMul"?gradient_tape/sequential_93/dense_289/ActivityRegularizer/Mul_1(1       @9       @A       @I       @a????P?i

s?=????Unknown
?oHostMul"Egradient_tape/sequential_93/dense_289/ActivityRegularizer/mul_1/Mul_1(1       @9       @A       @I       @a????P?iR?~????Unknown
?pHostReluGrad".gradient_tape/sequential_93/dense_289/ReluGrad(1       @9       @A       @I       @a????P?i1??????Unknown
?qHostSign"Bgradient_tape/sequential_93/dense_290/ActivityRegularizer/Abs/Sign(1       @9       @A       @I       @a????P?i?? ????Unknown
?rHostMul"=gradient_tape/sequential_93/dense_290/ActivityRegularizer/Mul(1       @9       @A       @I       @a????P?i???A???Unknown
?sHostMul"Cgradient_tape/sequential_93/dense_290/ActivityRegularizer/mul/Mul_1(1       @9       @A       @I       @a????P?i?̓????Unknown
?tHost
Reciprocal"Igradient_tape/sequential_93/dense_290/ActivityRegularizer/truediv/RealDiv(1       @9       @A       @I       @a????P?i???????Unknown
?uHostCast"0sequential_93/dense_288/ActivityRegularizer/Cast(1       @9       @A       @I       @a????P?i???Unknown
?vHostAddV2"1sequential_93/dense_288/ActivityRegularizer/add_1(1       @9       @A       @I       @a????P?i?j?E%???Unknown
?wHostMul"/sequential_93/dense_288/ActivityRegularizer/mul(1       @9       @A       @I       @a????P?i?I??-???Unknown
?xHostRealDiv"3sequential_93/dense_288/ActivityRegularizer/truediv(1       @9       @A       @I       @a????P?i?(??5???Unknown
?yHostReadVariableOp".sequential_93/dense_288/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????P?i??>???Unknown
?zHostAbs"/sequential_93/dense_289/ActivityRegularizer/Abs(1       @9       @A       @I       @a????P?i???IF???Unknown
?{HostCast"0sequential_93/dense_289/ActivityRegularizer/Cast(1       @9       @A       @I       @a????P?i?ł?N???Unknown
?|HostCast"0sequential_93/dense_290/ActivityRegularizer/Cast(1       @9       @A       @I       @a????P?iҤ??V???Unknown
?}HostSquare"2sequential_93/dense_290/ActivityRegularizer/Square(1       @9       @A       @I       @a????P?i΃?_???Unknown
?~HostSum"/sequential_93/dense_290/ActivityRegularizer/Sum(1       @9       @A       @I       @a????P?i?b?Mg???Unknown
?HostReadVariableOp"-sequential_93/dense_290/MatMul/ReadVariableOp(1       @9       @A       @I       @a????P?i?A??o???Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a????@?iD1?s???Unknown
Y?HostCast"Cast_4(1      ??9      ??A      ??I      ??a????@?i? ??w???Unknown
U?HostMul"Mul(1      ??9      ??A      ??I      ??a????@?i@?{???Unknown
}?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a????@?i???????Unknown
g?HostAddN"SGD/gradients/AddN_1(1      ??9      ??A      ??I      ??a????@?i<?1????Unknown
g?HostAddN"SGD/gradients/AddN_2(1      ??9      ??A      ??I      ??a????@?i?ށQ????Unknown
v?HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a????@?i8?r????Unknown
x?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????@?i????????Unknown
z?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????@?i4??????Unknown
??HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a????@?i???Ә???Unknown
y?HostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a????@?i0??????Unknown
??Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a????@?i?{?????Unknown
??HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a????@?i,k5????Unknown
??HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a????@?i?Z?U????Unknown
??HostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a????@?i(Jv????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a????@?i?9??????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a????@?i$)?????Unknown
??HostMul"Agradient_tape/sequential_93/dense_288/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a????@?i??׹???Unknown
??HostMul"?gradient_tape/sequential_93/dense_288/ActivityRegularizer/Mul_1(1      ??9      ??A      ??I      ??a????@?i ?????Unknown
??HostMul"Cgradient_tape/sequential_93/dense_288/ActivityRegularizer/mul/Mul_1(1      ??9      ??A      ??I      ??a????@?i???????Unknown
??HostMul"Agradient_tape/sequential_93/dense_289/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a????@?i? 9????Unknown
??HostMul"Agradient_tape/sequential_93/dense_290/ActivityRegularizer/Abs/mul(1      ??9      ??A      ??I      ??a????@?i?րY????Unknown
??HostMul"?gradient_tape/sequential_93/dense_290/ActivityRegularizer/Mul_1(1      ??9      ??A      ??I      ??a????@?i? z????Unknown
??HostMul"Egradient_tape/sequential_93/dense_290/ActivityRegularizer/mul_1/Mul_1(1      ??9      ??A      ??I      ??a????@?i????????Unknown
??HostSigmoidGrad"9gradient_tape/sequential_93/dense_290/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a????@?i? ?????Unknown
??HostMul"1sequential_93/dense_288/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a????@?i????????Unknown
??HostReadVariableOp"-sequential_93/dense_288/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a????@?i? ?????Unknown
??HostAddV2"1sequential_93/dense_289/ActivityRegularizer/add_1(1      ??9      ??A      ??I      ??a????@?i?s?????Unknown
??HostMul"/sequential_93/dense_289/ActivityRegularizer/mul(1      ??9      ??A      ??I      ??a????@?ic =????Unknown
??HostMul"1sequential_93/dense_289/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a????@?i?R?]????Unknown
??HostRealDiv"3sequential_93/dense_289/ActivityRegularizer/truediv(1      ??9      ??A      ??I      ??a????@?iB ~????Unknown
??HostAddV2"1sequential_93/dense_290/ActivityRegularizer/add_1(1      ??9      ??A      ??I      ??a????@?i?1??????Unknown
??HostMul"/sequential_93/dense_290/ActivityRegularizer/mul(1      ??9      ??A      ??I      ??a????@?i! ?????Unknown
??HostMul"1sequential_93/dense_290/ActivityRegularizer/mul_1(1      ??9      ??A      ??I      ??a????@?i???????Unknown
??HostRealDiv"3sequential_93/dense_290/ActivityRegularizer/truediv(1      ??9      ??A      ??I      ??a????@?i     ???Unknown
j?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
Z?HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
h?HostMul"Cgradient_tape/sequential_93/dense_289/ActivityRegularizer/mul/Mul_1(i     ???Unknown2CPU