"?Q
BHostIDLE"IDLE1     6?@A     6?@a?k?????i?k??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a?湞????i????Z@???Unknown?
iHostWriteSummary"WriteSummary(1      7@9      7@A      7@I      7@a??.*e?f?i?׾NW???Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a?:U\y?c?i?,??j???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      4@I      4@a?:U\y?c?i?wA?~???Unknown
vHost_FusedMatMul"sequential_99/dense_307/Relu(1      2@9      2@A      2@I      2@a??ӆ?a?i??J?e????Unknown
?HostMatMul",gradient_tape/sequential_99/dense_308/MatMul(1      1@9      1@A      1@I      1@aV?{???`?i*?U2????Unknown
?HostReadVariableOp"-sequential_99/dense_307/MatMul/ReadVariableOp(1      .@9      .@A      .@I      .@ay?
6?]?iW??????Unknown
v	Host_FusedMatMul"sequential_99/dense_308/Relu(1      .@9      .@A      .@I      .@ay?
6?]?i???׾???Unknown
?
HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      *@9      *@A      *@I      *@aGf?P?Y?i5?_4?????Unknown
`HostDivNoNan"
div_no_nan(1      (@9      (@A      (@I      (@a.??n^?W?i????????Unknown
THostMul"Mul(1      &@9      &@A      &@I      &@a???k?U?iʉ?j????Unknown
wHostCast"%gradient_tape/mean_squared_error/Cast(1      &@9      &@A      &@I      &@a???k?U?i??|OI????Unknown
dHostDataset"Iterator::Model(1      >@9      >@A      $@I      $@a?:U\y?S?i?*+????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?:U\y?S?i???????Unknown
?HostMatMul",gradient_tape/sequential_99/dense_309/MatMul(1      $@9      $@A      $@I      $@a?:U\y?S?iW???
???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A      "@I      "@a??ӆ?Q?i??H????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a??ӆ?Q?i?+Z????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@a??ӆ?Q?i???Ϝ%???Unknown
?HostMatMul".gradient_tape/sequential_99/dense_308/MatMul_1(1       @9       @A       @I       @a????(?O?i~?虄-???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a`D?C?K?i???jo4???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a`D?C?K?i?I?;Z;???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a`D?C?K?i???EB???Unknown
yHost_FusedMatMul"sequential_99/dense_309/BiasAdd(1      @9      @A      @I      @a`D?C?K?i??i?/I???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a`D?C?K?i?<J?P???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a.??n^?G?i????V???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a.??n^?G?i???]?[???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a.??n^?G?i'V5?a???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a.??n^?G?iR	??g???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a.??n^?G?i}?T??m???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a.??n^?G?i?o???s???Unknown
? HostMatMul",gradient_tape/sequential_99/dense_307/MatMul(1      @9      @A      @I      @a.??n^?G?i?"???y???Unknown
?!HostBiasAddGrad"9gradient_tape/sequential_99/dense_309/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a.??n^?G?i??'k????Unknown
?"HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a.??n^?G?i)??Bw????Unknown
X#HostCast"Cast_3(1      @9      @A      @I      @a?:U\y?C?ix?!h????Unknown
\$HostGreater"Greater(1      @9      @A      @I      @a?:U\y?C?iǳq?X????Unknown
V%HostSum"Sum_2(1      @9      @A      @I      @a?:U\y?C?i???I????Unknown
?&HostMatMul".gradient_tape/sequential_99/dense_309/MatMul_1(1      @9      @A      @I      @a?:U\y?C?ie??:????Unknown
u'HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a?:U\y?C?i??v?+????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a????(???i&k?????Unknown
?)HostBiasAddGrad"9gradient_tape/sequential_99/dense_307/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????(???i???d????Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_99/dense_308/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????(???i
Z?I????Unknown
?+HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a????(???i|??.?????Unknown
?,HostReadVariableOp".sequential_99/dense_309/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????(???i?H??????Unknown
t-HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a.??n^?7?i?"???????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a.??n^?7?i?n?ܷ???Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a.??n^?7?i??<?Ӻ???Unknown
?0HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      @I      @a.??n^?7?iF?
?ʽ???Unknown
}1HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @a.??n^?7?i܈خ?????Unknown
?2HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a.??n^?7?irb???????Unknown
?3HostReluGrad".gradient_tape/sequential_99/dense_308/ReluGrad(1      @9      @A      @I      @a.??n^?7?i<t??????Unknown
i4HostMean"mean_squared_error/Mean(1      @9      @A      @I      @a.??n^?7?i?Br?????Unknown
?5HostReadVariableOp"-sequential_99/dense_309/MatMul/ReadVariableOp(1      @9      @A      @I      @a.??n^?7?i4?^?????Unknown
t6HostSigmoid"sequential_99/dense_309/Sigmoid(1      @9      @A      @I      @a.??n^?7?i???I?????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a????(?/?i?g<?????Unknown
V8HostCast"Cast(1       @9       @A       @I       @a????(?/?i<@?.?????Unknown
X9HostCast"Cast_5(1       @9       @A       @I       @a????(?/?i?{y!?????Unknown
X:HostEqual"Equal(1       @9       @A       @I       @a????(?/?i??|????Unknown
V;HostMean"Mean(1       @9       @A       @I       @a????(?/?ig??v????Unknown
s<HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a????(?/?i /?o????Unknown
|=HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a????(?/?i?j??i????Unknown
u>HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a????(?/?i??'?c????Unknown
b?HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a????(?/?iK???]????Unknown
y@HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a????(?/?i:?W????Unknown
?AHostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a????(?/?i?YõQ????Unknown
uBHostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a????(?/?iv?L?K????Unknown
uCHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a????(?/?i/?՚E????Unknown
}DHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a????(?/?i?_??????Unknown
?EHostReluGrad".gradient_tape/sequential_99/dense_307/ReluGrad(1       @9       @A       @I       @a????(?/?i?H?9????Unknown
?FHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a????(?/?iZ?qr3????Unknown
|GHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a????(?/?i??d-????Unknown
?HHostReadVariableOp".sequential_99/dense_307/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a????(?/?i???W'????Unknown
?IHostReadVariableOp"-sequential_99/dense_308/MatMul/ReadVariableOp(1       @9       @A       @I       @a????(?/?i?7J!????Unknown
XJHostCast"Cast_4(1      ??9      ??A      ??I      ??a????(??ib?QC????Unknown
aKHostIdentity"Identity(1      ??9      ??A      ??I      ??a????(??i?s?<????Unknown?
?LHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a????(??i?5????Unknown
uMHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a????(??i??/????Unknown
wNHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a????(??i?Ld(????Unknown
wOHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a????(??i???!????Unknown
uPHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a????(??i???????Unknown
QHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a????(??im&2	????Unknown
wRHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a????(??iJ?v????Unknown
?SHostSigmoidGrad"9gradient_tape/sequential_99/dense_309/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a????(??i'b?????Unknown
?THostReadVariableOp".sequential_99/dense_308/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a????(??i     ???Unknown*?P
uHostFlushSummaryWriter"FlushSummaryWriter(1     8?@9     8?@A     8?@I     8?@a?6??Mm??i?6??Mm???Unknown?
iHostWriteSummary"WriteSummary(1      7@9      7@A      7@I      7@aa?5Xl??i2?v????Unknown?
sHostDataset"Iterator::Model::ParallelMapV2(1      4@9      4@A      4@I      4@a3A?L-??i<?܎!????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      4@I      4@a3A?L-??iF(B?P???Unknown
vHost_FusedMatMul"sequential_99/dense_307/Relu(1      2@9      2@A      2@I      2@a?E(B??iOmjS?????Unknown
?HostMatMul",gradient_tape/sequential_99/dense_308/MatMul(1      1@9      1@A      1@I      1@a?2A?L??i?t? ]???Unknown
?HostReadVariableOp"-sequential_99/dense_307/MatMul/ReadVariableOp(1      .@9      .@A      .@I      .@a?as?Ì?i^=@W????Unknown
vHost_FusedMatMul"sequential_99/dense_308/Relu(1      .@9      .@A      .@I      .@a?as?Ì?i?v?C???Unknown
?	HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1      *@9      *@A      *@I      *@a?!?c????ikS??Ԧ???Unknown
`
HostDivNoNan"
div_no_nan(1      (@9      (@A      (@I      (@ap?\???iq?\????Unknown
THostMul"Mul(1      &@9      &@A      &@I      &@aQ?_T???i? ]=@W???Unknown
wHostCast"%gradient_tape/mean_squared_error/Cast(1      &@9      &@A      &@I      &@aQ?_T???i{???????Unknown
dHostDataset"Iterator::Model(1      >@9      >@A      $@I      $@a3A?L-??i?Q?_T????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a3A?L-??i?"?E???Unknown
?HostMatMul",gradient_tape/sequential_99/dense_309/MatMul(1      $@9      $@A      $@I      $@a3A?L-??i??F⼑???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      .@9      .@A      "@I      "@a?E(B??i[??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?E(B??i?8o$????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      "@9      "@A      "@I      "@a?E(B??i[???`???Unknown
?HostMatMul".gradient_tape/sequential_99/dense_308/MatMul_1(1       @9       @A       @I       @a??z??~?i?x?3????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a??bk??z?i??O'?????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a??bk??z?i Z&??	???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??bk??z?i???G????Unknown
yHost_FusedMatMul"sequential_99/dense_309/BiasAdd(1      @9      @A      @I      @a??bk??z?i&??I?t???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a??bk??z?i?????????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @ap?\?w?i??bk?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ap?\?w?i??,????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\?w?i?????4???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\?w?i????b???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\?w?i?Cnǐ???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ap?\?w?i?4?.;???Unknown
?HostMatMul",gradient_tape/sequential_99/dense_307/MatMul(1      @9      @A      @I      @ap?\?w?i?K???????Unknown
? HostBiasAddGrad"9gradient_tape/sequential_99/dense_309/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ap?\?w?i?bk?????Unknown
?!HostCast"Smean_squared_error/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ap?\?w?i?y#q?H???Unknown
X"HostCast"Cast_3(1      @9      @A      @I      @a3A?L-s?iF⼑8o???Unknown
\#HostGreater"Greater(1      @9      @A      @I      @a3A?L-s?i?JV??????Unknown
V$HostSum"Sum_2(1      @9      @A      @I      @a3A?L-s?iJ????????Unknown
?%HostMatMul".gradient_tape/sequential_99/dense_309/MatMul_1(1      @9      @A      @I      @a3A?L-s?i???F????Unknown
u&HostSum"$mean_squared_error/weighted_loss/Sum(1      @9      @A      @I      @a3A?L-s?iN?"????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??z??n?iP>??O'???Unknown
?(HostBiasAddGrad"9gradient_tape/sequential_99/dense_307/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??z??n?iR??E???Unknown
?)HostBiasAddGrad"9gradient_tape/sequential_99/dense_308/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??z??n?iT????d???Unknown
?*HostSquaredDifference"$mean_squared_error/SquaredDifference(1      @9      @A      @I      @a??z??n?iVl[????Unknown
?+HostReadVariableOp".sequential_99/dense_309/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??z??n?iX&??	????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @ap?\?g?i?1?v????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @ap?\?g?iZ=@W????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @ap?\?g?i?H?7????Unknown
?/HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      @I      @ap?\?g?i\T?????Unknown
}0HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      @9      @A      @I      @ap?\?g?i?_T????Unknown
?1HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      @9      @A      @I      @ap?\?g?i^k??,???Unknown
?2HostReluGrad".gradient_tape/sequential_99/dense_308/ReluGrad(1      @9      @A      @I      @ap?\?g?i?v?C???Unknown
i3HostMean"mean_squared_error/Mean(1      @9      @A      @I      @ap?\?g?i`?h? Z???Unknown
?4HostReadVariableOp"-sequential_99/dense_309/MatMul/ReadVariableOp(1      @9      @A      @I      @ap?\?g?i???y#q???Unknown
t5HostSigmoid"sequential_99/dense_309/Sigmoid(1      @9      @A      @I      @ap?\?g?ib? Z&????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??z??^?ic?]?}????Unknown
V7HostCast"Cast(1       @9       @A       @I       @a??z??^?idS??Ԧ???Unknown
X8HostCast"Cast_5(1       @9       @A       @I       @a??z??^?ie??,????Unknown
X9HostEqual"Equal(1       @9       @A       @I       @a??z??^?if[?????Unknown
V:HostMean"Mean(1       @9       @A       @I       @a??z??^?igjS??????Unknown
s;HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a??z??^?ihǐ?1????Unknown
|<HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??z??^?ii$??????Unknown
u=HostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??z??^?ij?\????Unknown
b>HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a??z??^?ik?H?7???Unknown
y?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a??z??^?il;?܎!???Unknown
?@HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1       @9       @A       @I       @a??z??^?im???0???Unknown
uAHostMul"$gradient_tape/mean_squared_error/Mul(1       @9       @A       @I       @a??z??^?in? ]=@???Unknown
uBHostSub"$gradient_tape/mean_squared_error/sub(1       @9       @A       @I       @a??z??^?ioR>??O???Unknown
}CHostRealDiv"(gradient_tape/mean_squared_error/truediv(1       @9       @A       @I       @a??z??^?ip?{??^???Unknown
?DHostReluGrad".gradient_tape/sequential_99/dense_307/ReluGrad(1       @9       @A       @I       @a??z??^?iq?Cn???Unknown
?EHostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a??z??^?iri?]?}???Unknown
|FHostDivNoNan"&mean_squared_error/weighted_loss/value(1       @9       @A       @I       @a??z??^?is?3??????Unknown
?GHostReadVariableOp".sequential_99/dense_307/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??z??^?it#q?H????Unknown
?HHostReadVariableOp"-sequential_99/dense_308/MatMul/ReadVariableOp(1       @9       @A       @I       @a??z??^?iu???????Unknown
XIHostCast"Cast_4(1      ??9      ??A      ??I      ??a??z??N?i?.;K????Unknown
aJHostIdentity"Identity(1      ??9      ??A      ??I      ??a??z??N?iu??^?????Unknown?
?KHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      ??9      ??A      ??I      ??a??z??N?i??
??????Unknown
uLHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??z??N?iu:)?N????Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??z??N?i??G??????Unknown
wNHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a??z??N?iu?fߥ????Unknown
uOHostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a??z??N?i?E?Q????Unknown
PHostFloorDiv")gradient_tape/mean_squared_error/floordiv(1      ??9      ??A      ??I      ??a??z??N?iu???????Unknown
wQHostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??z??N?i??¿?????Unknown
?RHostSigmoidGrad"9gradient_tape/sequential_99/dense_309/Sigmoid/SigmoidGrad(1      ??9      ??A      ??I      ??a??z??N?iuQ?_T????Unknown
?SHostReadVariableOp".sequential_99/dense_308/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??z??N?i?????????Unknown2CPU