
public class PermIndFactory implements IndFactory {

    private final int n;

    public PermIndFactory(int n) {
        this.n = n;
    }

    @Override
    public PermInd getInstance() {
        return new PermInd(this.n);
    }
}
